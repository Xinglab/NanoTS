#!/usr/bin/env Rscript

#######
# Generic NanoTS-style benchmark script
#
# Required:
#   Rscript other_scripts/generic_benchmark.R \
#     --eval /path/to/NanoTS/eval.phase.txt \
#     --outdir benchmark_out
#
# Optional tool manifest:
#   Rscript other_scripts/generic_benchmark.R \
#     --eval /path/to/NanoTS/eval.phase.txt \
#     --tools tools.tsv \
#     --outdir benchmark_out \
#     --num-total 2 --num-alt 5 --ratio-alt 0.05
#
# tools.tsv columns:
#   tool        display name, e.g. NanoTS_phase, Clair3-RNA, LongCallR
#   file        result file path
#   type        nanots_eval, vcf, or table
# Optional columns:
#   pass_filter comma-separated PASS labels for VCF/table filter column
#   chrom_col,pos_col,ref_col,alt_col,filter_col,gt_col column names for table
#   sep         table separator, default tab
#   match_by    pos or allele, default pos
#######

suppressPackageStartupMessages({
  library(data.table)
})

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) {
    return(y)
  }
  x <- x[1]
  if (is.na(x) || x == "") y else x
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    help = FALSE,
    eval = "",
    tools = "",
    outdir = ".",
    num_total = 2,
    num_alt = 5,
    ratio_alt = 0.05,
    min_alt_ratio = 0,
    max_alt_ratio = 1,
    max_num_alt = NA_real_,
    max_num_total = NA_real_,
    target_chr = "",
    bed = "",
    thresholds = "",
    adjust_genotype = TRUE
  )

  i <- 1
  while (i <= length(args)) {
    key <- gsub("^--", "", args[i])
    key <- gsub("-", "_", key)
    value <- TRUE
    if (i < length(args) && !grepl("^--", args[i + 1])) {
      value <- args[i + 1]
      i <- i + 1
    }
    out[[key]] <- value
    i <- i + 1
  }

  numeric_keys <- c("num_total", "num_alt", "ratio_alt", "min_alt_ratio", "max_alt_ratio",
                    "max_num_alt", "max_num_total")
  for (key in numeric_keys) {
    if (!is.null(out[[key]]) && !is.na(out[[key]]) && out[[key]] != "") {
      out[[key]] <- as.numeric(out[[key]])
    }
  }
  out$adjust_genotype <- !(tolower(as.character(out$adjust_genotype)) %in% c("false", "f", "0", "no"))
  out
}

print_usage <- function() {
  cat("
Usage:
  Rscript other_scripts/generic_benchmark.R --eval eval.phase.txt --outdir benchmark_out

Optional:
  --tools tools.tsv
  --num-total 2
  --num-alt 5
  --ratio-alt 0.05
  --thresholds 2,5,10
  --target-chr chr20
  --bed regions.bed.gz

tools.tsv required columns:
  tool, file, type

Supported type values:
  nanots_eval   NanoTS eval.txt or eval.phase.txt
  vcf           VCF/VCF.gz from LongCallR, Clair3-RNA, DeepVariant, etc.
  table         Generic table with configurable columns

Example tools.tsv:
tool\tfile\ttype\tpass_filter
Clair3-RNA\t/path/output.vcf.gz\tvcf\tALL
LongCallR\t/path/lc.vcf\tvcf\tnot:HomRef
DeepVariant\t/path/output.vcf.gz\tvcf\tnot:HomRef

")
}

stop_if_missing <- function(path, label) {
  if (is.null(path) || is.na(path) || path == "" || !file.exists(path)) {
    stop(label, " does not exist: ", path, call. = FALSE)
  }
}

safe_div <- function(x, y) {
  ifelse(is.na(y) | y == 0, 0, x / y)
}

calculate_f1 <- function(precision, recall) {
  if (is.na(precision) || is.na(recall) || (precision == 0 && recall == 0)) {
    return(0)
  }
  2 * precision * recall / (precision + recall)
}

read_eval_table <- function(eval_file, target_chr = "", bed_file = "") {
  stop_if_missing(eval_file, "eval file")
  evaldf <- fread(eval_file)
  colnames(evaldf)[1:2] <- c("chrom", "pos")
  evaldf <- evaldf[chrom != "chrom"]
  evaldf[, pos := as.integer(pos)]

  if (!"ratio_1" %in% colnames(evaldf)) {
    evaldf[, ratio_1 := safe_div(as.numeric(alt_reads), as.numeric(ref_reads) + as.numeric(alt_reads))]
  }

  if (target_chr != "") {
    evaldf <- evaldf[chrom == target_chr]
  }

  if (bed_file != "") {
    stop_if_missing(bed_file, "BED file")
    bed_df <- fread(cmd = paste("zcat -f", shQuote(bed_file)), select = c(1, 2, 3),
                    col.names = c("chrom", "start", "end"))
    bed_df[, `:=`(start = as.integer(start), end = as.integer(end))]
    evaldf[, end := pos]
    setkey(bed_df, chrom, start, end)
    evaldf <- foverlaps(evaldf, bed_df, by.x = c("chrom", "pos", "end"),
                        by.y = c("chrom", "start", "end"), nomatch = 0)
  }

  evaldf[, key := paste(chrom, pos, sep = "|")]
  if (!"GIAB_ALT" %in% colnames(evaldf)) {
    evaldf[, GIAB_ALT := alt]
  }
  evaldf[, skey := paste(key, ref, alt, sep = "|")]
  evaldf <- evaldf[!duplicated(key)]
  evaldf[, SNV_zygosity := 0L]
  evaldf[Zygosity == "Het" & Label == "ALT", SNV_zygosity := 1L]
  evaldf[Zygosity == "Hom" & Label == "ALT", SNV_zygosity := 2L]
  evaldf[]
}

callable_subset <- function(evaldf, num_total, num_alt, ratio_alt,
                            min_alt_ratio = 0, max_alt_ratio = 1,
                            max_num_alt = NA, max_num_total = NA) {
  x <- evaldf[(as.numeric(ref_reads) + as.numeric(alt_reads)) >= num_total &
                as.numeric(alt_reads) >= num_alt &
                as.numeric(ratio_1) >= ratio_alt &
                as.numeric(ratio_1) >= min_alt_ratio &
                as.numeric(ratio_1) <= max_alt_ratio]
  if (!is.na(max_num_alt)) {
    x <- x[as.numeric(alt_reads) <= max_num_alt]
  }
  if (!is.na(max_num_total)) {
    x <- x[(as.numeric(ref_reads) + as.numeric(alt_reads)) <= max_num_total]
  }
  x
}

estimate_performance <- function(evaldf, tool, num_total = 2, num_alt = 5, ratio_alt = 0.05,
                                 min_alt_ratio = 0, max_alt_ratio = 1,
                                 max_num_alt = NA, max_num_total = NA) {
  callregion <- callable_subset(evaldf, num_total, num_alt, ratio_alt,
                                min_alt_ratio, max_alt_ratio, max_num_alt, max_num_total)
  pass_df <- callregion[filter == "PASS"]
  alt_df <- callregion[Label == "ALT"]

  precision <- safe_div(sum(pass_df$Label == "ALT"), nrow(pass_df))
  recall <- safe_div(sum(alt_df$filter == "PASS"), nrow(alt_df))
  f1 <- calculate_f1(precision, recall)

  out <- data.table(
    tool = tool,
    num_total = num_total,
    num_alt = num_alt,
    ratio_alt = ratio_alt,
    min_alt_ratio = min_alt_ratio,
    max_alt_ratio = max_alt_ratio,
    Precision = precision,
    Recall = recall,
    F1 = f1,
    identified_SNPs_by_caller = nrow(pass_df),
    gold_standard_SNPs = nrow(alt_df),
    identified_gold_standard_SNPs_by_caller = sum(pass_df$Label == "ALT")
  )
  setnames(
    out,
    c("identified_SNPs_by_caller", "gold_standard_SNPs", "identified_gold_standard_SNPs_by_caller"),
    c("identified SNPs by caller", "gold-standard SNPs", "Identified gold-standard SNPs by caller")
  )
  out
}

estimate_zygosity <- function(evaldf, tool, num_total = 2, num_alt = 5, ratio_alt = 0.05,
                              min_alt_ratio = 0, max_alt_ratio = 1,
                              max_num_alt = NA, max_num_total = NA) {
  callregion <- callable_subset(evaldf, num_total, num_alt, ratio_alt,
                                min_alt_ratio, max_alt_ratio, max_num_alt, max_num_total)
  pred <- factor(callregion$filter_zyg, levels = c(0, 1, 2))
  truth <- factor(callregion$SNV_zygosity, levels = c(0, 1, 2))
  conf <- table(pred, truth)

  precision <- recall <- f1 <- numeric(3)
  support <- colSums(conf)
  for (i in seq_len(3)) {
    precision[i] <- safe_div(conf[i, i], sum(conf[i, ]))
    recall[i] <- safe_div(conf[i, i], sum(conf[, i]))
    f1[i] <- calculate_f1(precision[i], recall[i])
  }
  data.table(
    tool = tool,
    num_total = num_total,
    num_alt = num_alt,
    Precision_G0 = precision[1],
    Precision_G1 = precision[2],
    Precision_G2 = precision[3],
    Recall_G0 = recall[1],
    Recall_G1 = recall[2],
    Recall_G2 = recall[3],
    F1_G0 = f1[1],
    F1_G1 = f1[2],
    F1_G2 = f1[3],
    Precision_macro = mean(precision),
    Recall_macro = mean(recall),
    F1_macro = mean(f1),
    F1_weighted = safe_div(sum(f1 * support), sum(support))
  )
}

zygosity_wide_to_long <- function(zygosity_df) {
  if (nrow(zygosity_df) == 0) {
    return(data.table())
  }
  rbindlist(list(
    zygosity_df[, .(
      tool, num_total, num_alt,
      genotype = "0/0",
      Precision = Precision_G0,
      Recall = Recall_G0,
      F1 = F1_G0
    )],
    zygosity_df[, .(
      tool, num_total, num_alt,
      genotype = "0/1",
      Precision = Precision_G1,
      Recall = Recall_G1,
      F1 = F1_G1
    )],
    zygosity_df[, .(
      tool, num_total, num_alt,
      genotype = "1/1",
      Precision = Precision_G2,
      Recall = Recall_G2,
      F1 = F1_G2
    )]
  ), use.names = TRUE)
}

adjust_genotype <- function(evaldf) {
  if (!("GIAB_ALT" %in% colnames(evaldf)) || !("alt" %in% colnames(evaldf))) {
    return(evaldf)
  }
  evaldf[filter == "PASS" & alt != GIAB_ALT & GIAB_ALT != ".", filter := "no"]
  evaldf
}

empty_tool_manifest <- function() {
  data.table(tool = character(), file = character(), type = character())
}

read_tool_manifest <- function(tools_file) {
  if (tools_file == "") {
    return(empty_tool_manifest())
  }
  stop_if_missing(tools_file, "tools manifest")
  tools <- fread(tools_file)
  required <- c("tool", "file", "type")
  missing <- setdiff(required, colnames(tools))
  if (length(missing) > 0) {
    stop("tools manifest is missing required columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  tools
}

parse_vcf_calls <- function(file, pass_filter = "") {
  stop_if_missing(file, "VCF file")
  vcf <- fread(file, header = FALSE, comment.char = "#", fill = TRUE)
  if (nrow(vcf) == 0) {
    out <- data.table(.site_key = character(), ref = character(), alt = character(), filter = character(), gt = character())
    setnames(out, ".site_key", "key")
    return(out)
  }
  colnames(vcf)[1:min(10, ncol(vcf))] <- paste0("V", 1:min(10, ncol(vcf)))
  pass_filter <- pass_filter %||% "PASS,RNAEditing,."
  vcf[, key := paste(V1, V2, sep = "|")]
  if (toupper(pass_filter) %in% c("ALL", "ANY", "*")) {
    vcf[, filter_call := "PASS"]
  } else if (startsWith(pass_filter, "not:")) {
    fail_set <- unlist(strsplit(sub("^not:", "", pass_filter), ","))
    vcf[, filter_call := ifelse(!(V7 %in% fail_set), "PASS", "no")]
  } else {
    pass_set <- unlist(strsplit(pass_filter, ","))
    vcf[, filter_call := ifelse(V7 %in% pass_set, "PASS", "no")]
  }
  gt <- if ("V10" %in% colnames(vcf)) vcf$V10 else rep("", nrow(vcf))
  out <- data.table(.site_key = vcf$key, ref = vcf$V4, alt = vcf$V5, filter = vcf$filter_call, gt = gt)
  setnames(out, ".site_key", "key")
  out
}

parse_table_calls <- function(row) {
  file <- row$file
  stop_if_missing(file, "tool table")
  sep <- row$sep %||% "\t"
  x <- fread(file, sep = sep)
  chrom_col <- row$chrom_col %||% "chrom"
  pos_col <- row$pos_col %||% "pos"
  ref_col <- row$ref_col %||% "ref"
  alt_col <- row$alt_col %||% "alt"
  filter_col <- row$filter_col %||% "filter"
  gt_col <- row$gt_col %||% ""

  needed <- c(chrom_col, pos_col)
  missing <- setdiff(needed, colnames(x))
  if (length(missing) > 0) {
    stop("table tool ", row$tool, " is missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  pass_set <- unlist(strsplit(row$pass_filter %||% "PASS", ","))
  filter_value <- if (filter_col %in% colnames(x)) x[[filter_col]] else "PASS"
  gt <- if (gt_col %in% colnames(x)) x[[gt_col]] else ""
  ref <- if (ref_col %in% colnames(x)) x[[ref_col]] else "."
  alt <- if (alt_col %in% colnames(x)) x[[alt_col]] else "."

  out <- data.table(
    .site_key = paste(x[[chrom_col]], x[[pos_col]], sep = "|"),
    ref = ref,
    alt = alt,
    filter = ifelse(filter_value %in% pass_set, "PASS", "no"),
    gt = gt
  )
  setnames(out, ".site_key", "key")
  out
}

parse_nanots_eval_calls <- function(file) {
  x <- read_eval_table(file)
  out <- data.table(
    .site_key = x$key,
    ref = x$ref,
    alt = x$alt,
    filter = ifelse(as.numeric(x$Result_snv) > 0, "PASS", "no"),
    gt = ifelse(as.numeric(x$Result_snv) > 0,
                ifelse(as.numeric(x$Result_heterzygosity) > 0, "0/1", "1/1"),
                "0/0")
  )
  setnames(out, ".site_key", "key")
  out
}

gt_to_zygosity <- function(gt) {
  gt <- as.character(gt)
  out <- rep(0L, length(gt))
  out[grepl("0[/|]1|1[/|]0", gt)] <- 1L
  out[grepl("1[/|]1", gt)] <- 2L
  out
}

apply_tool_calls <- function(base_eval, tool_row, adjust = TRUE) {
  type <- tolower(tool_row$type)
  calls <- switch(
    type,
    nanots_eval = parse_nanots_eval_calls(tool_row$file),
    vcf = parse_vcf_calls(tool_row$file, tool_row$pass_filter %||% ""),
    table = parse_table_calls(tool_row),
    stop("Unsupported tool type: ", tool_row$type, call. = FALSE)
  )
  calls <- calls[!duplicated(key)]

  eval_tool <- copy(base_eval)
  eval_tool[, `:=`(filter = "no", filter_zyg = 0L)]
  idx <- match(calls$key, eval_tool$key)
  keep <- which(!is.na(idx))
  if (length(keep) > 0) {
    eval_tool[idx[keep], filter := calls$filter[keep]]
    eval_tool[idx[keep], alt := calls$alt[keep]]
    eval_tool[idx[keep], filter_zyg := gt_to_zygosity(calls$gt[keep])]
  }
  if (adjust) {
    eval_tool <- adjust_genotype(eval_tool)
    eval_tool[filter != "PASS", filter_zyg := 0L]
  }
  eval_tool
}

make_default_tools <- function(eval_file) {
  data.table(
    tool = "NanoTS",
    file = eval_file,
    type = "nanots_eval",
    pass_filter = "",
    match_by = "pos"
  )
}

main <- function() {
  args <- parse_args()
  if (isTRUE(args$help)) {
    print_usage()
    quit(status = 0)
  }
  stop_if_missing(args$eval, "NanoTS eval benchmark file")
  dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)

  base_eval <- read_eval_table(args$eval, target_chr = args$target_chr, bed_file = args$bed)
  tools <- rbindlist(list(make_default_tools(args$eval), read_tool_manifest(args$tools)), fill = TRUE)
  tools <- tools[file.exists(file)]
  if (nrow(tools) == 0) {
    stop("No readable tool files found.", call. = FALSE)
  }

  threshold_values <- if (args$thresholds != "") {
    as.numeric(unlist(strsplit(args$thresholds, ",")))
  } else {
    args$num_alt
  }

  summary_list <- list()
  zygosity_list <- list()
  callable_list <- list()

  for (i in seq_len(nrow(tools))) {
    tool_row <- tools[i]
    message("Benchmarking ", tool_row$tool, ": ", tool_row$file)
    eval_tool <- apply_tool_calls(base_eval, tool_row, adjust = args$adjust_genotype)

    for (threshold in threshold_values) {
      num_total <- if (args$thresholds != "") threshold else args$num_total
      num_alt <- threshold
      summary_list[[length(summary_list) + 1]] <- estimate_performance(
        eval_tool, tool_row$tool, num_total, num_alt, args$ratio_alt,
        args$min_alt_ratio, args$max_alt_ratio, args$max_num_alt, args$max_num_total
      )
      zygosity_list[[length(zygosity_list) + 1]] <- estimate_zygosity(
        eval_tool, tool_row$tool, num_total, num_alt, args$ratio_alt,
        args$min_alt_ratio, args$max_alt_ratio, args$max_num_alt, args$max_num_total
      )
    }

    callable_list[[length(callable_list) + 1]] <- data.table(
      tool = tool_row$tool,
      n_sites = nrow(eval_tool),
      n_pass = sum(eval_tool$filter == "PASS"),
      n_truth_alt = sum(eval_tool$Label == "ALT")
    )
  }

  summary_df <- rbindlist(summary_list, fill = TRUE)
  zygosity_df <- rbindlist(zygosity_list, fill = TRUE)
  genotype_df <- zygosity_wide_to_long(zygosity_df)
  callable_df <- rbindlist(callable_list, fill = TRUE)

  fwrite(summary_df, file.path(args$outdir, "summary_metrics.tsv"), sep = "\t")
  fwrite(genotype_df, file.path(args$outdir, "genotype_metrics.tsv"), sep = "\t")
  fwrite(callable_df, file.path(args$outdir, "tool_site_counts.tsv"), sep = "\t")
  message("Wrote: ", file.path(args$outdir, "summary_metrics.tsv"))
  message("Wrote: ", file.path(args$outdir, "genotype_metrics.tsv"))
  message("Wrote: ", file.path(args$outdir, "tool_site_counts.tsv"))
}

main()
