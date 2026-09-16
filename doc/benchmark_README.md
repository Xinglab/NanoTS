# NanoTS Benchmark Workflow

This document describes how to reproduce the existing `evaluate_predict_HG.py` eval tables and the metrics from `Rcode/functions.R` (`get_metric_all_zyg()` and `get_metric_all_zyg_deepvariant()`). Metrics are computed on the NanoTS candidate background after the specified read-depth and VAF filters.

The important rule is that **NanoTS builds the eval table**. NanoTS keeps the full candidate background, while other tools may only output called sites. Other tools should be compared later by overlaying their calls onto the NanoTS eval background.

## Inputs

Required files:

- NanoTS prediction/detail table, including non-SNP rows
  - Usually `unphased_predict.txt` or `phased_predict.txt`; do not substitute a PASS-only VCF
- Truth VCF, for example GIAB high-confidence VCF
- Optional confident BED region

Scripts (paths relative to the NanoTS repository root):

```text
other_scripts/generate_eval_table.py
other_scripts/generic_benchmark.R
```

Run all commands below from the **NanoTS repository root**, with the Python environment activated and `Rscript` available on `PATH`. Replace `/path/to/...` with your actual paths. Create the parent directory of each Python `--output` file first; the R benchmark creates its `--outdir` automatically. Keep chromosome names consistent across the prediction table, truth VCF, BED, and caller files (for example, `chr1` throughout).

The Python generator needs `pandas` and `pysam`. BED filtering is built in and accepts plain or gzipped BED files. The R benchmark needs `data.table`.

## Step 1: Generate NanoTS Eval Table

If your NanoTS prediction table already contains candidate and prediction columns:

```bash
python \
  other_scripts/generate_eval_table.py \
  --nanots /path/to/unphased_predict.txt \
  --truth-vcf /path/to/GIAB.vcf.gz \
  --bed /path/to/confident_regions.bed.gz \
  --output /path/to/eval.txt
```

If you want the eval background to come from a separate NanoTS candidate table, add `--candidates`. Prediction columns from `--nanots` are overlaid onto the candidate rows by `chrom,pos,ref,alt`; candidate rows without a prediction are kept as non-SNP calls.

```bash
python \
  other_scripts/generate_eval_table.py \
  --candidates /path/to/unphased_alt.txt \
  --nanots /path/to/unphased_predict.txt \
  --truth-vcf /path/to/GIAB.vcf.gz \
  --bed /path/to/confident_regions.bed.gz \
  --output /path/to/eval.txt
```

For phased prediction:

```bash
python \
  other_scripts/generate_eval_table.py \
  --nanots /path/to/phased_predict.txt \
  --truth-vcf /path/to/GIAB.vcf.gz \
  --bed /path/to/confident_regions.bed.gz \
  --output /path/to/eval.phase.txt
```

The default output preserves every input column and row order, replaces missing input values with zero, and appends only `Label`, `Zygosity`, and `GIAB_ALT`, exactly as the original generator does. Prediction columns are not synthesized unless `--candidates` is used. A typical prediction table produces:

```text
chrom pos ref alt ref_reads alt_reads ratio_1
genotype_0 genotype_1 genotype_2
Result_snv Result_heterzygosity Result_genotype
Label Zygosity GIAB_ALT
```

### Reproduce the original command

The original option names are aliases, so the same command can use the new script:

```bash
python other_scripts/generate_eval_table.py \
  --lr_variants /path/to/unphased_predict.txt \
  --bed_file /path/to/confident_regions.bed \
  --hc_vcf /path/to/GIAB.vcf.gz \
  --chrom chr1 \
  --long_vcf_in /path/to/eval.chr1
```

Use the same chromosome, BED, indexed truth VCF, prediction table, and input separator for byte-identical output. Repeating `--chrom chr1 --chrom chr2 ...` concatenates chromosomes in the supplied order, matching the original shell loop with a single header. Without `--chrom`, input row order is retained across chromosomes. `--threads` does not change output order or contents.

For historical equivalence, keep default position matching and do not enable `--candidates`, `--truth-table`, `--match-by allele`, `--keep-outside-bed`, or `--drop-truth-indel-region`; those are extensions without an original equivalent. The original does not exclude indel regions. The output is always tab-separated, even when `--sep` specifies a different input separator.

## Step 2: Prepare Optional Tool Manifest

The R benchmark always includes NanoTS from `--eval`. Optional tools can be added with a manifest TSV.

Example `tools.tsv` (tab-separated, with literal tab characters between fields):

```text
tool	file	type	pass_filter
Clair3-RNA	/path/to/clair3rna.vcf.gz	vcf	ALL
LongCallR	/path/to/longcallr.vcf	vcf	not:HomRef
DeepVariant	/path/to/deepvariant.vcf.gz	vcf	not:HomRef
```

If no `--tools` is provided, the benchmark reports NanoTS only. The names `Clair3-RNA`, `LongCallR` (also `LongcallR`), and `DeepVariant` select the corresponding historical caller conventions. When using another display name, add a `legacy_caller` column with one of these names. Missing tool files produce an error.

For exact historical DeepVariant reproduction, the manifest must also contain one LongCallR VCF. The original DeepVariant wrapper uses the LongCallR positions after restricting LongCallR to the eval background when constructing DeepVariant keys. This behavior is intentionally preserved, including the original R vector recycling; retain the original input file ordering. It is a historical compatibility rule, not a recommended independent DeepVariant matching method.

## Step 3: Final Precision, Recall, F1

Run one benchmark at ALT >= 5 and total depth >= 2, matching the default in the original `get_metric_all_zyg()` wrapper:

```bash
Rscript other_scripts/generic_benchmark.R \
  --eval /path/to/eval.phase.txt \
  --tools /path/to/tools.tsv \
  --outdir /path/to/benchmark_ALT5 \
  --num-total 2 \
  --num-alt 5 \
  --ratio-alt 0.05
```

Main outputs:

```text
/path/to/benchmark_ALT5/summary_metrics.tsv
/path/to/benchmark_ALT5/genotype_metrics.tsv
/path/to/benchmark_ALT5/genotype_metrics_wide.tsv
/path/to/benchmark_ALT5/tool_site_counts.tsv
```

`summary_metrics.tsv` reports SNV Precision, Recall, and F1 on the filtered candidate background. Empty precision/recall denominators remain `NaN`, matching the original R function; the corresponding F1 is zero.

Important columns:

```text
tool Precision Recall F1 identified SNPs by caller gold-standard SNPs Identified gold-standard SNPs by caller
```

`genotype_metrics.tsv` reports genotype-level Precision, Recall, and F1 for each genotype class. `genotype_metrics_wide.tsv` also exports all 21 genotype metrics returned by the original function, including macro, weighted, and variant-only metrics. The TSV reports retain the generic wrapper layout; metric values reproduce the original returned matrices. `tool_site_counts.tsv` counts sites after chromosome/BED/tie preprocessing and call overlay, before depth and VAF thresholds; it is not a per-threshold metric table.

Example columns:

```text
tool num_total num_alt genotype Precision Recall F1
NanoTS 2 5 0/0 ... ... ...
NanoTS 2 5 0/1 ... ... ...
NanoTS 2 5 1/1 ... ... ...
```


## ALT-Read Threshold Series

Use `--thresholds` to compute multiple ALT-read thresholds in one run. In this mode, each threshold is used for both `num_total` and `num_alt`.

```bash
Rscript other_scripts/generic_benchmark.R \
  --eval /path/to/eval.phase.txt \
  --tools /path/to/tools.tsv \
  --outdir /path/to/benchmark_ALT_thresholds \
  --thresholds 2,3,4,5,10 \
  --ratio-alt 0.05
```

Outputs:

```text
/path/to/benchmark_ALT_thresholds/summary_metrics.tsv
/path/to/benchmark_ALT_thresholds/genotype_metrics.tsv
```

Example rows:

```text
tool	num_total	num_alt	Precision	Recall	F1
NanoTS	2	2	...
NanoTS	3	3	...
NanoTS	5	5	...
```

## VAF Bin Series

For VAF bins, run the benchmark once per VAF interval using `--min-alt-ratio` and `--max-alt-ratio`. Both limits are inclusive to reproduce the original R code: a site at VAF 0.20 appears in both neighboring bins. Do not sum bin counts as if the bins were disjoint.

Example bins:

```text
0.05-0.20
0.20-0.35
0.35-0.50
0.50-0.65
0.65-0.80
0.80-1.00
```

Bash loop:

```bash
EVAL=/path/to/eval.phase.txt
TOOLS=/path/to/tools.tsv
OUTROOT=/path/to/benchmark_VAF_bins

mkdir -p "${OUTROOT}"

for BIN in 0.05:0.20 0.20:0.35 0.35:0.50 0.50:0.65 0.65:0.80 0.80:1.00
do
  MIN_VAF=${BIN%%:*}
  MAX_VAF=${BIN##*:}
  LABEL="${MIN_VAF}-${MAX_VAF}"

  Rscript other_scripts/generic_benchmark.R \
    --eval "${EVAL}" \
    --tools "${TOOLS}" \
    --outdir "${OUTROOT}/${LABEL}" \
    --num-total 2 \
    --num-alt 5 \
    --ratio-alt 0.05 \
    --min-alt-ratio "${MIN_VAF}" \
    --max-alt-ratio "${MAX_VAF}"
done
```

Collect all VAF-bin summary tables:

```bash
{
  first=1
  for FILE in /path/to/benchmark_VAF_bins/*/summary_metrics.tsv
  do
    BIN=$(basename "$(dirname "${FILE}")")
    if [ "${first}" -eq 1 ]; then
      awk -v bin="${BIN}" 'BEGIN{FS=OFS="\t"} NR==1{print "VAF_bin",$0} NR>1{print bin,$0}' "${FILE}"
      first=0
    else
      awk -v bin="${BIN}" 'BEGIN{FS=OFS="\t"} NR>1{print bin,$0}' "${FILE}"
    fi
  done
} > /path/to/benchmark_VAF_bins/summary_metrics.by_VAF.tsv
```

Collect all VAF-bin genotype metrics:

```bash
{
  first=1
  for FILE in /path/to/benchmark_VAF_bins/*/genotype_metrics.tsv
  do
    BIN=$(basename "$(dirname "${FILE}")")
    if [ "${first}" -eq 1 ]; then
      awk -v bin="${BIN}" 'BEGIN{FS=OFS="\t"} NR==1{print "VAF_bin",$0} NR>1{print bin,$0}' "${FILE}"
      first=0
    else
      awk -v bin="${BIN}" 'BEGIN{FS=OFS="\t"} NR>1{print bin,$0}' "${FILE}"
    fi
  done
} > /path/to/benchmark_VAF_bins/genotype_metrics.by_VAF.tsv
```

## Notes

- `--ratio-alt 0.05` is the callable-site lower VAF cutoff used in your existing R code.
- `--num-alt` is the minimum ALT read threshold.
- `--num-total` is the minimum total read threshold.
- `--thresholds` uses the same value for `num-alt` and `num-total`.
- The eval table should be generated once from NanoTS and reused for comparisons.
- Other tools should not build the eval background because they may omit non-called candidate sites.

## Historical conventions retained

- Truth matches `(chrom, pos)`, without checking the called ALT or truth genotype for ALT presence. The last truth ALT at a position is saved in `GIAB_ALT`. A site is `Het` if any sample in any record at that position has GT `(0,1)` or `(1,0)`; other genotypes are `Hom`.
- Python confidence BED filtering uses `start + 1 <= pos <= end`. Optional R `--bed` filtering retains the original R inclusive `start <= pos <= end` convention. These two stages intentionally differ.
- R keeps the first eval row per position. Passing external calls are overlaid in input order, so the last passing ALT at a position wins.
- SNV allele adjustment changes wrong-ALT calls to non-PASS before calculating precision. Genotype metrics independently use the original predicted genotypes, including filtered or allele-mismatched calls.
- LongCallR and DeepVariant recognize unphased `1/1` as homozygous ALT; Clair3-RNA also recognizes `1|1`, as in the reference branches. Genotype matching uses the full first sample field, as the original does.
- Both VAF limits are inclusive. Adjacent example bins therefore share sites at their boundaries.
- `--remove-tie` and `--only-tie` reproduce the options in `get_metric_all_zyg()` and require `alt_tie` in the eval table. The original DeepVariant wrapper has no tie options.
- The original DeepVariant wrapper defaults to `num_total=5, num_alt=5`; pass `--num-total 5 --num-alt 5` to reproduce those defaults. The generic CLI defaults to the three-caller wrapper's `num_total=2, num_alt=5`.
