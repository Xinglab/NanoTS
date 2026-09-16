# NanoTS Benchmark Workflow

This document describes how to build a NanoTS-based eval table and compute final Precision, Recall, and F1 across ALT-read thresholds and VAF bins.

The important rule is that **NanoTS builds the eval table**. NanoTS keeps the full candidate background, while other tools may only output called sites. Other tools should be compared later by overlaying their calls onto the NanoTS eval background.

## Inputs

Required files:

- NanoTS prediction/detail table
  - Usually `unphased_predict.txt` or `phased_predict.txt`
- Truth VCF, for example GIAB high-confidence VCF
- Optional confident BED region

Scripts:

```bash
NanoTS/other_scripts/generate_eval_table.py
NanoTS/other_scripts/generic_benchmark.R
```

Run the Python command inside an environment with NanoTS dependencies installed.

BED filtering in `generate_eval_table.py` uses `pybedtools`, so users must install both `pybedtools` and the `bedtools` command-line tools before running commands with `--bed`.

Example:

```bash
conda install -c bioconda pybedtools bedtools
```

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

The output contains the fields used by the benchmark code:

```text
chrom pos ref alt ref_reads alt_reads ratio_1
genotype_0 genotype_1 genotype_2
Result_snv Result_heterzygosity Result_genotype
Label Zygosity GIAB_ALT
```

## Step 2: Prepare Optional Tool Manifest

The R benchmark always includes NanoTS from `--eval`. Optional tools can be added with a manifest TSV.

Example `tools.tsv`:

```text
tool	file	type	pass_filter
Clair3-RNA	/path/to/clair3rna.vcf.gz	vcf	ALL
LongCallR	/path/to/longcallr.vcf	vcf	not:HomRef
DeepVariant	/path/to/deepvariant.vcf.gz	vcf	not:HomRef
```

If no `--tools` is provided, the benchmark reports NanoTS only.

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
```

`summary_metrics.tsv` reports overall SNV Precision, Recall, and F1.

Important columns:

```text
tool Precision Recall F1 identified SNPs by caller gold-standard SNPs Identified gold-standard SNPs by caller
```

`genotype_metrics.tsv` reports genotype-level Precision, Recall, and F1 for each genotype class.

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

For VAF bins, run the benchmark once per VAF interval using `--min-alt-ratio` and `--max-alt-ratio`.

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
