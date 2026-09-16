# Zenodo 19900025: NanoTS benchmark results

## Run the benchmark

Download and extract `GIAB_variant_calling_results.tar.gz` from [Zenodo](https://zenodo.org/records/19900025). Run the commands below from the NanoTS repository root, with `Rscript` and the R package `data.table` installed. Set `DATA_DIR` to the extracted data directory and `OUTROOT` to your output directory.

This evaluates phased and unphased NanoTS for all eight datasets, alongside Clair3-RNA and LongcallR-nf, using total depth >= 2, ALT reads >= 5, and VAF >= 0.05. Each comparison uses the corresponding NanoTS eval background and the script's default allele adjustment.

**VCF FILTER selection (`V7`):** Clair3-RNA uses `ALL`, which includes all three FILTER types observed in these files: `PASS`, `LowQual`, and `RNAEditing`. LongcallR-nf uses `not:HomRef`, which includes its `PASS` and `LowQual` records and excludes records whose FILTER is exactly `HomRef`. Thus, low-quality calls are included for both callers. These settings reproduce the original benchmark's filtering rules; the NanoTS candidate background, depth/VAF thresholds, and default allele adjustment still apply when calculating SNV precision, recall, and F1.

```bash
DATA_DIR=/path/to/GIAB_variant_calling_results
OUTROOT=/path/to/benchmark_results

for sample_dir in "$DATA_DIR"/*/*
do
  platform=$(basename "$(dirname "$sample_dir")")
  sample=$(basename "$sample_dir")
  sample_out="$OUTROOT/$platform/$sample"
  mkdir -p "$sample_out"

  tools="$sample_out/tools.tsv"
  printf 'tool\tfile\ttype\tpass_filter\n' > "$tools"
  printf 'Clair3-RNA\t%s\tvcf\tALL\n' \
    "$sample_dir/Clair3RNA/output_enable_phasing.vcf.gz" >> "$tools"
  printf 'LongCallR\t%s\tvcf\tnot:HomRef\n' \
    "$sample_dir/LongcallR_nf/lc_nf.vcf" >> "$tools"

  for stage in phased unphased
  do
    eval_file="$sample_dir/NanoTS/eval.$stage.txt"
    # The archive uses eval.uphased.txt for ONT_dRNA/HG004.
    if [ "$stage" = unphased ] && [ ! -f "$eval_file" ]; then
      eval_file="$sample_dir/NanoTS/eval.uphased.txt"
    fi

    Rscript other_scripts/generic_benchmark.R \
      --eval "$eval_file" \
      --tools "$tools" \
      --outdir "$sample_out/$stage" \
      --num-total 2 \
      --num-alt 5 \
      --ratio-alt 0.05
  done
done
```

Precision, recall, and F1 are written to `$OUTROOT/<platform>/<sample>/<stage>/summary_metrics.tsv`. Results below are percentages rounded to two decimals.

## Phased NanoTS background

| Platform | Sample | Caller | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---:|---:|---:|
| ONT_cDNA | HG001 | NanoTS | 98.83 | 97.39 | 98.11 |
| ONT_cDNA | HG001 | Clair3-RNA | 89.65 | 97.52 | 93.42 |
| ONT_cDNA | HG001 | LongcallR-nf | 93.45 | 78.60 | 85.38 |
| ONT_cDNA | HG004 | NanoTS | 98.32 | 95.66 | 96.97 |
| ONT_cDNA | HG004 | Clair3-RNA | 82.26 | 95.62 | 88.44 |
| ONT_cDNA | HG004 | LongcallR-nf | 86.21 | 76.57 | 81.10 |
| ONT_cDNA | HG005 | NanoTS | 98.36 | 96.98 | 97.66 |
| ONT_cDNA | HG005 | Clair3-RNA | 83.29 | 97.00 | 89.62 |
| ONT_cDNA | HG005 | LongcallR-nf | 87.27 | 78.69 | 82.76 |
| ONT_cDNA_TEQUILA_IEI | HG001 | NanoTS | 97.53 | 95.70 | 96.60 |
| ONT_cDNA_TEQUILA_IEI | HG001 | Clair3-RNA | 80.79 | 96.18 | 87.81 |
| ONT_cDNA_TEQUILA_IEI | HG001 | LongcallR-nf | 86.68 | 75.80 | 80.88 |
| ONT_dRNA | HG004 | NanoTS | 98.16 | 97.93 | 98.04 |
| ONT_dRNA | HG004 | Clair3-RNA | 95.75 | 99.06 | 97.38 |
| ONT_dRNA | HG004 | LongcallR-nf | 99.04 | 90.21 | 94.42 |
| ONT_dRNA | HG005 | NanoTS | 98.15 | 98.00 | 98.07 |
| ONT_dRNA | HG005 | Clair3-RNA | 95.67 | 99.20 | 97.40 |
| ONT_dRNA | HG005 | LongcallR-nf | 98.97 | 90.11 | 94.33 |
| PacBio_MASseq | HG004 | NanoTS | 99.37 | 98.07 | 98.72 |
| PacBio_MASseq | HG004 | Clair3-RNA | 99.42 | 98.06 | 98.74 |
| PacBio_MASseq | HG004 | LongcallR-nf | 99.10 | 96.29 | 97.67 |
| PacBio_MASseq | HG005 | NanoTS | 99.35 | 98.25 | 98.80 |
| PacBio_MASseq | HG005 | Clair3-RNA | 99.37 | 98.34 | 98.85 |
| PacBio_MASseq | HG005 | LongcallR-nf | 99.05 | 96.55 | 97.78 |

## Unphased NanoTS background

| Platform | Sample | Caller | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---:|---:|---:|
| ONT_cDNA | HG001 | NanoTS | 98.55 | 97.12 | 97.83 |
| ONT_cDNA | HG001 | Clair3-RNA | 89.65 | 97.52 | 93.42 |
| ONT_cDNA | HG001 | LongcallR-nf | 93.45 | 78.60 | 85.38 |
| ONT_cDNA | HG004 | NanoTS | 97.57 | 95.60 | 96.57 |
| ONT_cDNA | HG004 | Clair3-RNA | 82.26 | 95.62 | 88.44 |
| ONT_cDNA | HG004 | LongcallR-nf | 86.21 | 76.57 | 81.10 |
| ONT_cDNA | HG005 | NanoTS | 97.67 | 96.81 | 97.24 |
| ONT_cDNA | HG005 | Clair3-RNA | 83.29 | 97.00 | 89.62 |
| ONT_cDNA | HG005 | LongcallR-nf | 87.27 | 78.69 | 82.76 |
| ONT_cDNA_TEQUILA_IEI | HG001 | NanoTS | 97.12 | 94.71 | 95.90 |
| ONT_cDNA_TEQUILA_IEI | HG001 | Clair3-RNA | 80.79 | 96.18 | 87.81 |
| ONT_cDNA_TEQUILA_IEI | HG001 | LongcallR-nf | 86.68 | 75.80 | 80.88 |
| ONT_dRNA | HG004 | NanoTS | 97.44 | 97.90 | 97.67 |
| ONT_dRNA | HG004 | Clair3-RNA | 95.75 | 99.06 | 97.38 |
| ONT_dRNA | HG004 | LongcallR-nf | 99.04 | 90.21 | 94.42 |
| ONT_dRNA | HG005 | NanoTS | 97.43 | 97.94 | 97.68 |
| ONT_dRNA | HG005 | Clair3-RNA | 95.67 | 99.20 | 97.40 |
| ONT_dRNA | HG005 | LongcallR-nf | 98.97 | 90.11 | 94.33 |
| PacBio_MASseq | HG004 | NanoTS | 99.32 | 97.42 | 98.36 |
| PacBio_MASseq | HG004 | Clair3-RNA | 99.42 | 98.06 | 98.74 |
| PacBio_MASseq | HG004 | LongcallR-nf | 99.10 | 96.29 | 97.67 |
| PacBio_MASseq | HG005 | NanoTS | 99.24 | 97.65 | 98.44 |
| PacBio_MASseq | HG005 | Clair3-RNA | 99.37 | 98.34 | 98.85 |
| PacBio_MASseq | HG005 | LongcallR-nf | 99.05 | 96.55 | 97.78 |
