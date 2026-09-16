#!/usr/bin/env python3
"""
Generate a NanoTS-style eval table.

Only NanoTS should build the eval table, because NanoTS keeps the full
candidate universe. Other callers can be benchmarked later by overlaying their
calls onto this eval table; they should not define the eval background.

This script follows the older script/evaluate_predict_HG.py pattern:
  1. read NanoTS prediction/detail table
  2. optionally restrict to high-confidence BED regions
  3. annotate each NanoTS candidate with truth Label, Zygosity, GIAB_ALT
  4. keep/add NanoTS prediction columns used by the R benchmark code
"""

import argparse
import sys
from multiprocessing import Pool

import pandas as pd
import progressbar
import pysam

try:
    import pybedtools
except ImportError:
    pybedtools = None


NANOTS_RESULT_COLUMNS = [
    "genotype_0",
    "genotype_1",
    "genotype_2",
    "Result_snv",
    "Result_heterzygosity",
    "Result_genotype",
]

WORKER_TRUTH = None
WORKER_INDEL_POSITIONS = None
WORKER_MATCH_BY = None
WORKER_COLUMNS = None
WORKER_OUTPUT_COLUMNS = None
WORKER_SEP = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create eval.txt/eval.phase.txt from a NanoTS prediction/detail table plus truth labels.",
        epilog="Dependency note: --bed filtering requires pybedtools and the bedtools command-line tools."
    )
    parser.add_argument(
        "--nanots",
        required=True,
        help="NanoTS prediction/detail table. Usually unphased_predict.txt or phased_predict.txt.",
    )
    parser.add_argument(
        "--candidates",
        help="Optional NanoTS candidate table, such as unphased_alt.txt. "
             "When provided, this file defines the eval background and --nanots predictions are overlaid by chrom,pos,ref,alt.",
    )
    parser.add_argument(
        "--truth-vcf",
        help="Truth VCF/VCF.gz, e.g. GIAB high-confidence calls. Used for Label, Zygosity, GIAB_ALT.",
    )
    parser.add_argument(
        "--truth-table",
        help="Truth TSV instead of truth VCF. Columns: chrom,pos,ref,alt and optional GT or Zygosity.",
    )
    parser.add_argument("--bed", help="Optional BED/BED.gz high-confidence regions. Sites outside BED are skipped.")
    parser.add_argument("--chrom", action="append", help="Optional chromosome to keep. Can be used multiple times.")
    parser.add_argument("--sep", default="\t", help="Separator for NanoTS/truth tables (default: tab).")
    parser.add_argument("--output", required=True, help="Output eval table.")
    parser.add_argument(
        "--match-by",
        choices=["pos", "allele"],
        default="pos",
        help="Truth matching key. Your R code usually uses position matching (default: pos).",
    )
    parser.add_argument(
        "--keep-outside-bed",
        action="store_true",
        help="Keep sites outside --bed instead of dropping them.",
    )
    parser.add_argument(
        "--drop-truth-indel-region",
        action="store_true",
        help="Drop NanoTS sites inside truth indel regions.",
    )
    parser.add_argument("--threads", type=int, default=1, help="Number of worker processes for row annotation.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress messages.")
    return parser.parse_args()


def log(message, quiet=False):
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def make_progress_bar(label, total, quiet=False):
    if quiet:
        return None
    widgets = [label, ': ', progressbar.Percentage(), ' ',
               progressbar.Bar(marker='=', left='[', right=']'), ' ',
               progressbar.ETA()]
    return progressbar.ProgressBar(widgets=widgets, maxval=total).start()


def require_pybedtools():
    if pybedtools is None:
        raise SystemExit(
            "pybedtools is required when --bed is used. Install it before running this code, for example: "
            "conda install -c bioconda pybedtools bedtools or pip install pybedtools with bedtools available in PATH."
        )


def variant_key(chrom, pos, ref=None, alt=None, match_by="pos"):
    if match_by == "allele":
        return (str(chrom), int(pos), str(ref), str(alt))
    return (str(chrom), int(pos))


def normalize_gt(gt):
    if gt is None:
        return "./."
    if isinstance(gt, tuple):
        if any(x is None for x in gt):
            return "./."
        return "/".join(str(x) for x in gt)
    return str(gt).split(":")[0]


def gt_to_zygosity(gt):
    gt = normalize_gt(gt)
    alleles = gt.replace("|", "/").split("/")
    if len(alleles) < 2 or "." in alleles:
        return "Hom"
    if alleles[0] != alleles[1]:
        return "Het"
    return "Hom"


def filter_nanots_by_bed(nanots, bed_path):
    require_pybedtools()
    if nanots.empty:
        return nanots.copy()

    point_df = pd.DataFrame({
        "chrom": nanots["chrom"].astype(str),
        "start": nanots["pos"].astype(int) - 1,
        "end": nanots["pos"].astype(int),
        "row_id": range(len(nanots)),
    })
    candidate_bed = pybedtools.BedTool.from_dataframe(point_df)
    confident_bed = pybedtools.BedTool(bed_path)
    overlap = candidate_bed.intersect(confident_bed, u=True)
    keep_idx = [int(feature.fields[3]) for feature in overlap]

    keep = [False] * len(nanots)
    for idx in keep_idx:
        keep[idx] = True
    return nanots.loc[keep].copy()


def update_truth_record(truth, key, alt, zygosity):
    old = truth.get(key)
    if old is None or (old["Zygosity"] != "Het" and zygosity == "Het"):
        truth[key] = {"GIAB_ALT": alt, "Zygosity": zygosity}


def add_truth_record(truth, indel_positions, rec, match_by="pos"):
    if rec.alts is None:
        return
    sample_gt = None
    if rec.samples:
        first_sample = next(iter(rec.samples.values()))
        sample_gt = first_sample.get("GT")
    zygosity = gt_to_zygosity(sample_gt)
    for alt in rec.alts:
        if len(rec.ref) > 1 or len(alt) > 1:
            for pos in range(rec.pos, rec.pos + max(len(rec.ref), len(alt)) + 1):
                indel_positions.add((rec.chrom, pos))
        key = variant_key(rec.chrom, rec.pos, rec.ref, alt, match_by)
        update_truth_record(truth, key, alt, zygosity)
        if match_by == "pos":
            pos_key = variant_key(rec.chrom, rec.pos, match_by="pos")
            update_truth_record(truth, pos_key, alt, zygosity)


def read_truth_vcf(vcf_path, match_by="pos", chrom_filter=None):
    truth = {}
    indel_positions = set()
    chrom_filter = set(chrom_filter) if chrom_filter else None
    with pysam.VariantFile(vcf_path) as vcf:
        if chrom_filter:
            try:
                for chrom in chrom_filter:
                    records = vcf.fetch(chrom)
                    for rec in records:
                        add_truth_record(truth, indel_positions, rec, match_by)
                return truth, indel_positions
            except (ValueError, OSError):
                pass
        for rec in vcf:
            if chrom_filter and rec.chrom not in chrom_filter:
                continue
            add_truth_record(truth, indel_positions, rec, match_by)
    return truth, indel_positions


def annotate_row(row, truth, indel_positions, match_by):
    key = variant_key(row["chrom"], row["pos"], row["ref"], row["alt"], match_by)
    truth_info = truth.get(key)
    if truth_info is None and match_by == "allele":
        truth_info = truth.get(variant_key(row["chrom"], row["pos"], match_by="pos"))
    if truth_info is None:
        row["Label"] = "REF"
        row["Zygosity"] = "Hom"
        row["GIAB_ALT"] = "."
    else:
        row["Label"] = "ALT"
        row["Zygosity"] = truth_info["Zygosity"]
        row["GIAB_ALT"] = truth_info["GIAB_ALT"]
    row["truth_indel_region"] = (row["chrom"], int(row["pos"])) in indel_positions
    return row


def format_annotated_row(row, output_columns, sep):
    return sep.join(str(row[col]) for col in output_columns)


def init_worker(truth, indel_positions, match_by, columns, output_columns, sep):
    global WORKER_TRUTH
    global WORKER_INDEL_POSITIONS
    global WORKER_MATCH_BY
    global WORKER_COLUMNS
    global WORKER_OUTPUT_COLUMNS
    global WORKER_SEP
    WORKER_TRUTH = truth
    WORKER_INDEL_POSITIONS = indel_positions
    WORKER_MATCH_BY = match_by
    WORKER_COLUMNS = columns
    WORKER_OUTPUT_COLUMNS = output_columns
    WORKER_SEP = sep


def annotate_row_worker(row_tuple):
    row = dict(zip(WORKER_COLUMNS, row_tuple))
    row = annotate_row(row, WORKER_TRUTH, WORKER_INDEL_POSITIONS, WORKER_MATCH_BY)
    return row["truth_indel_region"], row["Label"], format_annotated_row(row, WORKER_OUTPUT_COLUMNS, WORKER_SEP)


def annotate_row_local(row_tuple, columns, output_columns, truth, indel_positions, match_by, sep):
    row = dict(zip(columns, row_tuple))
    row = annotate_row(row, truth, indel_positions, match_by)
    return row["truth_indel_region"], row["Label"], format_annotated_row(row, output_columns, sep)


def write_eval_rows(nanots, output, truth, indel_positions, match_by, drop_truth_indel_region=False,
                    sep="\t", threads=1, quiet=False):
    output_columns = list(nanots.columns) + ["Label", "Zygosity", "GIAB_ALT", "truth_indel_region"]
    total = len(nanots)
    written = 0
    truth_alt = 0
    columns = list(nanots.columns)
    bar = make_progress_bar("Annotate/write NanoTS candidates", total, quiet) if total > 0 else None

    with open(output, "w") as handle:
        handle.write(sep.join(output_columns) + "\n")
        if threads > 1 and total > 0:
            rows = nanots.itertuples(index=False, name=None)
            chunksize = max(1, min(10000, total // (threads * 4) if total >= threads * 4 else 1))
            with Pool(
                processes=threads,
                initializer=init_worker,
                initargs=(truth, indel_positions, match_by, columns, output_columns, sep),
            ) as pool:
                for idx, (truth_indel_region, label, line) in enumerate(pool.imap(annotate_row_worker, rows, chunksize=chunksize), 1):
                    if not (drop_truth_indel_region and truth_indel_region):
                        handle.write(line + "\n")
                        written += 1
                        if label == "ALT":
                            truth_alt += 1
                    if bar is not None:
                        bar.update(idx)
        else:
            for idx, row_tuple in enumerate(nanots.itertuples(index=False, name=None), 1):
                truth_indel_region, label, line = annotate_row_local(
                    row_tuple, columns, output_columns, truth, indel_positions, match_by, sep
                )
                if not (drop_truth_indel_region and truth_indel_region):
                    handle.write(line + "\n")
                    written += 1
                    if label == "ALT":
                        truth_alt += 1
                if bar is not None:
                    bar.update(idx)
    if bar is not None:
        bar.finish()
    return written, truth_alt


def read_truth_table(path, sep="\t", match_by="pos"):
    df = pd.read_csv(path, sep=sep)
    truth = {}
    required = {"chrom", "pos", "alt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Truth table missing columns: {','.join(sorted(missing))}")
    for _, row in df.iterrows():
        gt = row["GT"] if "GT" in df.columns else ""
        zygosity = row["Zygosity"] if "Zygosity" in df.columns else gt_to_zygosity(gt)
        key = variant_key(row["chrom"], row["pos"], row.get("ref", "."), row["alt"], match_by)
        truth[key] = {"GIAB_ALT": row["alt"], "Zygosity": zygosity}
    return truth, set()


def read_variant_table(path, sep="\t", label="variant table"):
    df = pd.read_csv(path, sep=sep, comment="#").fillna(0)
    required = {"chrom", "pos", "ref", "alt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {','.join(sorted(missing))}")
    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = df["pos"].astype(int)
    return df


def read_nanots_table(path, sep="\t"):
    return read_variant_table(path, sep, "NanoTS table")


def ensure_nanots_result_columns(df):
    defaults = {
        "genotype_0": 1.0,
        "genotype_1": 0.0,
        "genotype_2": 0.0,
        "Result_snv": 0,
        "Result_heterzygosity": 0,
        "Result_genotype": 0,
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
        else:
            df[col] = df[col].fillna(value)
    return df


def overlay_prediction_on_candidates(candidates, prediction):
    key_cols = ["chrom", "pos", "ref", "alt"]
    prediction = prediction.drop_duplicates(subset=key_cols).copy()
    overlay_cols = [col for col in prediction.columns if col not in key_cols and col not in candidates.columns]
    for col in NANOTS_RESULT_COLUMNS:
        if col in prediction.columns and col not in overlay_cols and col not in key_cols:
            overlay_cols.append(col)
    if not overlay_cols:
        merged = candidates.copy()
    else:
        merged = candidates.merge(prediction[key_cols + overlay_cols], on=key_cols, how="left")
    merged = ensure_nanots_result_columns(merged)
    return merged.fillna(0)


def main():
    args = parse_args()
    if not args.truth_vcf and not args.truth_table:
        raise ValueError("One of --truth-vcf or --truth-table is required")

    chrom_filter = set(args.chrom) if args.chrom else None
    log(f"Reading NanoTS prediction table: {args.nanots}", args.quiet)
    prediction = read_nanots_table(args.nanots, args.sep)
    prediction = ensure_nanots_result_columns(prediction)
    if args.candidates:
        log(f"Reading NanoTS candidate table: {args.candidates}", args.quiet)
        candidates = read_variant_table(args.candidates, args.sep, "NanoTS candidate table")
        nanots = overlay_prediction_on_candidates(candidates, prediction)
        log(f"Candidate rows loaded: {len(candidates)}", args.quiet)
        log(f"Prediction rows loaded: {len(prediction)}", args.quiet)
    else:
        nanots = prediction
    log(f"NanoTS rows loaded: {len(nanots)}", args.quiet)

    if chrom_filter:
        nanots = nanots[nanots["chrom"].isin(chrom_filter)].copy()
        log(f"Rows after chromosome filter: {len(nanots)}", args.quiet)

    if args.bed and not args.keep_outside_bed:
        log("Using pybedtools for BED filtering", args.quiet)
    else:
        log("No BED filter provided" if not args.bed else "Keeping sites outside BED", args.quiet)
    if args.bed and not args.keep_outside_bed:
        log("Applying BED filter", args.quiet)
        nanots = filter_nanots_by_bed(nanots, args.bed)
        log(f"Rows after BED filter: {len(nanots)}", args.quiet)

    if args.truth_vcf:
        log(f"Reading truth VCF: {args.truth_vcf}", args.quiet)
        truth, indel_positions = read_truth_vcf(args.truth_vcf, args.match_by, chrom_filter)
    else:
        log(f"Reading truth table: {args.truth_table}", args.quiet)
        truth, indel_positions = read_truth_table(args.truth_table, args.sep, args.match_by)
    log(f"Truth records loaded: {len(truth)}", args.quiet)
    log(f"Truth indel positions loaded: {len(indel_positions)}", args.quiet)

    log("Sorting eval table", args.quiet)
    nanots = nanots.sort_values(["chrom", "pos"]).reset_index(drop=True)
    written, truth_alt = write_eval_rows(
        nanots,
        args.output,
        truth,
        indel_positions,
        args.match_by,
        args.drop_truth_indel_region,
        args.sep,
        max(1, args.threads),
        args.quiet,
    )
    log(f"Wrote eval table: {args.output}", args.quiet)
    log(f"Rows: {written}", args.quiet)
    log(f"Truth ALT rows: {truth_alt}", args.quiet)


if __name__ == "__main__":
    main()
