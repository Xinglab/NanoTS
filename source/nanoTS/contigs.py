"""Select reference contigs and name their intermediate files consistently."""

import csv
import re
from urllib.parse import quote

import pysam


DEFAULT_CONTIGS = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX", "chrY")


def add_contigs_argument(parser):
    parser.add_argument(
        "--contigs", nargs="+", metavar="CONTIG",
        help="Contigs to analyze, using exact BAM/FASTA names. Use 'auto' for all "
             "shared contigs. Default: chr1-chr22, chrX, chrY; --region selects its contig.",
    )


def parse_region(region, lengths):
    """Return an exact contig name and inclusive, one-based coordinates."""
    if region in lengths:
        return region, 1, lengths[region]
    match = re.fullmatch(r"(.+):(\d+)-(\d+)", region)
    if match is None:
        raise ValueError(f"Unknown contig or invalid region {region!r}; use an exact "
                         "BAM contig name or 'contig:start-end'.")
    chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if chrom not in lengths:
        raise ValueError(f"Region contig {chrom!r} is not in the BAM header.")
    if start < 1 or start > end or start > lengths[chrom]:
        raise ValueError(f"Invalid coordinates in region {region!r}.")
    return chrom, start, min(end, lengths[chrom])


def resolve_contigs(bam_path, ref_path, contigs=None, region=None):
    """Validate a selection against BAM and FASTA names and sequence lengths."""
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            bam_lengths = dict(zip(bam.references, bam.lengths))
        with pysam.FastaFile(ref_path) as reference:
            ref_lengths = dict(zip(reference.references, reference.lengths))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read BAM/reference contigs: {exc}") from exc

    requested = ([contigs] if isinstance(contigs, str) else list(contigs)) if contigs is not None else None
    if requested is not None and not requested:
        raise ValueError("--contigs requires 'auto' or at least one contig name.")
    if requested and "auto" in requested and requested != ["auto"]:
        raise ValueError("Use --contigs auto by itself, without additional names.")

    region_chrom = parse_region(region, bam_lengths)[0] if region else None
    if requested == ["auto"]:
        selected = [chrom for chrom in bam_lengths if chrom in ref_lengths]
    elif requested is not None:
        selected = list(dict.fromkeys(requested))
    elif region_chrom:
        selected = [region_chrom]
    else:
        selected = [chrom for chrom in DEFAULT_CONTIGS if chrom in bam_lengths]

    if not selected:
        raise ValueError("No matching target contigs found in the BAM and reference. "
                         "For non-chr names, use --contigs auto or --contigs with exact names.")
    for chrom in selected:
        if chrom not in bam_lengths:
            raise ValueError(f"Selected contig {chrom!r} is not in the BAM header.")
        if chrom not in ref_lengths:
            raise ValueError(f"Selected contig {chrom!r} is not in the reference FASTA.")
    if region_chrom:
        if region_chrom not in selected:
            raise ValueError(f"Region contig {region_chrom!r} is excluded by --contigs.")
        selected = [region_chrom]
    for chrom in selected:
        if bam_lengths[chrom] != ref_lengths[chrom]:
            raise ValueError(f"Length mismatch for {chrom!r}: BAM={bam_lengths[chrom]}, "
                             f"FASTA={ref_lengths[chrom]}.")
    return selected


def table_contigs(path):
    """Read exact contig names from a NanoTS prediction table without coercion."""
    with open(path) as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or "chrom" not in reader.fieldnames:
            raise ValueError(f"Prediction table {str(path)!r} has no 'chrom' column.")
        names = set()
        for row in reader:
            if not row.get("chrom"):
                raise ValueError(f"Missing contig name in prediction table {str(path)!r}.")
            names.add(row["chrom"])
    return names


def contig_suffix(chrom):
    """Keep .chr1 filenames while safely encoding other reference names."""
    return "." + quote(str(chrom), safe="._-")


def prediction_contigs(target_chr, header_contigs):
    """Accept exact names and the numeric/X/Y tokens used by existing callers."""
    selected = []
    for name in target_chr:
        name = str(name)
        if name not in header_contigs and f"chr{name}" in header_contigs:
            name = f"chr{name}"
        if name not in header_contigs:
            raise ValueError(f"Prediction contig {name!r} is not in the BAM/VCF header.")
        selected.append(name)
    return selected
