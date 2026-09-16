# NanoTS Benchmark Workflow

This workflow generates NanoTS eval tables and calculates precision, recall, and F1 for NanoTS and other variant callers. Metrics are computed on the NanoTS candidate background after the specified read-depth and VAF filters.

Use the same **NanoTS candidate background** for all callers in a comparison. Build the eval table from NanoTS candidate and prediction rows, including non-SNP rows, then overlay the other callers' calls onto that table.

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

For a NanoTS prediction table containing candidate and prediction columns:

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

The default output preserves input columns and row order, replaces missing input values with zero, and appends `Label`, `Zygosity`, and `GIAB_ALT`. Prediction columns are not synthesized unless `--candidates` is used. A typical prediction table produces:

```text
chrom pos ref alt ref_reads alt_reads ratio_1
genotype_0 genotype_1 genotype_2
Result_snv Result_heterzygosity Result_genotype
Label Zygosity GIAB_ALT
```

### Select a chromosome

Use `--chrom` to restrict the eval table to a chromosome:

```bash
python other_scripts/generate_eval_table.py \
  --nanots /path/to/unphased_predict.txt \
  --bed /path/to/confident_regions.bed \
  --truth-vcf /path/to/GIAB.vcf.gz \
  --chrom chr1 \
  --output /path/to/eval.chr1
```

Repeating `--chrom chr1 --chrom chr2 ...` concatenates chromosomes in the supplied order with a single header. Without `--chrom`, input row order is retained across chromosomes. Use `--threads` to set the number of annotation workers; it does not change output order or contents.

Truth matching uses chromosome and position by default; `--match-by allele` also requires matching REF and ALT alleles. Truth indel regions are kept unless `--drop-truth-indel-region` is specified. The output is always tab-separated; `--sep` controls the input-table separator.

## Step 2: Prepare Optional Tool Manifest

The R benchmark always includes NanoTS from `--eval`. Optional tools can be added with a manifest TSV.

Example `tools.tsv` (tab-separated, with literal tab characters between fields):

```text
tool	file	type	pass_filter
Clair3-RNA	/path/to/clair3rna.vcf.gz	vcf	ALL
LongCallR	/path/to/longcallr.vcf	vcf	not:HomRef
DeepVariant	/path/to/deepvariant.vcf.gz	vcf	not:HomRef
```

If no `--tools` is provided, the benchmark reports NanoTS only. The names `Clair3-RNA`, `LongCallR` (also `LongcallR`), and `DeepVariant` select caller-specific filtering and genotype handling. When using another display name, add a `legacy_caller` column with one of these names. Missing tool files produce an error.

DeepVariant evaluation requires exactly one LongCallR VCF in the manifest. The script constructs DeepVariant site keys using DeepVariant chromosome names and LongCallR positions after restricting LongCallR to the eval background. Positions are paired by row, with R vector recycling when lengths differ, so the results depend on the LongCallR input and row ordering.

## Step 3: Final Precision, Recall, F1

Run a benchmark with the default thresholds: ALT reads >= 5, total depth >= 2, and VAF >= 0.05:

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

`summary_metrics.tsv` reports SNV Precision, Recall, and F1 on the filtered candidate background. Precision or recall is `NaN` when its denominator is zero; the corresponding F1 is zero.

Important columns:

```text
tool Precision Recall F1 identified SNPs by caller gold-standard SNPs Identified gold-standard SNPs by caller
```

`genotype_metrics.tsv` reports precision, recall, and F1 for each genotype class. `genotype_metrics_wide.tsv` contains 21 genotype metrics, including per-class, macro, weighted, and variant-only metrics. `tool_site_counts.tsv` counts sites after chromosome/BED/tie preprocessing and call overlay, before depth and VAF thresholds; it is not a per-threshold metric table.

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

For VAF bins, run the benchmark once per VAF interval using `--min-alt-ratio` and `--max-alt-ratio`. Both limits are inclusive: a site at VAF 0.20 appears in both neighboring bins. Do not sum bin counts as if the bins were disjoint.

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

- `--ratio-alt` sets the minimum VAF for callable sites; the default is `0.05`.
- `--num-alt` is the minimum ALT read threshold.
- `--num-total` is the minimum total read threshold.
- `--thresholds` uses the same value for `num-alt` and `num-total`.
- The eval table should be generated once from NanoTS and reused for comparisons.
- Other tools should not build the eval background because they may omit non-called candidate sites.
