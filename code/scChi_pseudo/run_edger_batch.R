#!/usr/bin/env Rscript

## ------------------------------------------------------------
## edgeR on ATAC pseudobulk counts
## expects files like:
##   scATAC_pseudobulk_counts.csv
##   scATAC_pseudobulk_metadata.csv
## ------------------------------------------------------------

# ----- ensure and register a user-writable library -----
userlib <- Sys.getenv("R_LIBS_USER")
if (userlib == "") {
  userlib <- file.path(
    Sys.getenv("HOME"),
    "R",
    paste0(getRversion()$major, ".", getRversion()$minor),
    "library"
  )
  Sys.setenv(R_LIBS_USER = userlib)
}
dir.create(userlib, recursive = TRUE, showWarnings = FALSE)
.libPaths(unique(c(userlib, .libPaths())))

# ----- install/load packages into user library if needed -----
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages(
    "BiocManager",
    repos = "https://cloud.r-project.org",
    lib = userlib
  )
  .libPaths(unique(c(userlib, .libPaths())))
}
if (!requireNamespace("edgeR", quietly = TRUE) ||
    !requireNamespace("limma", quietly = TRUE)) {
  BiocManager::install(
    c("edgeR", "limma"),
    ask = FALSE,
    update = FALSE,
    lib = userlib
  )
  .libPaths(unique(c(userlib, .libPaths())))
}

suppressPackageStartupMessages({
  library(edgeR, lib.loc = userlib)
  library(limma,  lib.loc = userlib)
})

## ------------------------------------------------------------
## Find all *_pseudobulk_counts.csv files in this folder
## ------------------------------------------------------------
dir <- "."
cfs <- list.files(dir,
                  pattern = "_pseudobulk_counts\\.csv$",
                  full.names = TRUE)

if (length(cfs) == 0L) {
  stop("No *_pseudobulk_counts.csv files found in ", dir)
}

for (cf in cfs) {
  base <- sub("_counts\\.csv$", "", cf)
  colf <- paste0(base, "_metadata.csv")

  message("\n=== ", basename(base), " ===")

  if (!file.exists(colf)) {
    message("  ! Metadata file not found: ", colf, "  → skipping")
    next
  }

  ## -------------------- READ COUNTS -------------------------
  counts_df <- read.csv(cf, check.names = FALSE)
  if (!"Sample_ID" %in% colnames(counts_df)) {
    stop("Counts file missing 'Sample_ID' column: ", cf)
  }
  rownames(counts_df) <- counts_df$Sample_ID
  counts_df$Sample_ID <- NULL

  # genes as rows, samples as columns
  counts_mat <- t(as.matrix(counts_df))
  storage.mode(counts_mat) <- "double"
  counts <- counts_mat

  ## -------------------- READ METADATA -----------------------
  coldata <- read.csv(colf, row.names = 1, check.names = FALSE)

  # align columns (samples)
  common <- intersect(colnames(counts), rownames(coldata))
  message("  Matched ", length(common), " samples between counts and metadata")
  if (length(common) < 4L) {
    message("  ! Skipping (too few matched samples)")
    next
  }
  counts  <- counts[, common, drop = FALSE]
  coldata <- coldata[ common, , drop = FALSE]

  ## -------------------- CONDITION FACTOR --------------------
  # If 'condition' already exists, use it; otherwise build from Response_Status
  if ("condition" %in% colnames(coldata)) {
    coldata$condition <- factor(coldata$condition)
  } else if ("Response_Status" %in% colnames(coldata)) {
    # Aggressive vs everything else as Control
    coldata$condition <- ifelse(coldata$Response_Status == "Aggressive",
                                "Aggressive", "Control")
    coldata$condition <- factor(coldata$condition,
                                levels = c("Control", "Aggressive"))
  } else {
    message("  ! No 'condition' or 'Response_Status' column; skipping")
    next
  }

  ## -------------------- BATCH FACTOR ------------------------
  if ("Batch_ID" %in% colnames(coldata)) {
    coldata$batch <- factor(coldata$Batch_ID)
  }

  ## -------------------- DROP ZERO-LIB SAMPLES ---------------
  libsizes <- colSums(counts, na.rm = TRUE)
  message("  Library sizes (first 10): ",
          paste(round(head(libsizes, 10), 4), collapse = ", "))
  keep_lib <- libsizes > 0
  message("  Samples with non-zero library size: ",
          sum(keep_lib), " of ", length(keep_lib))

  if (sum(keep_lib) < 4L) {
    message("  ! Too few samples with non-zero library size; skipping")
    next
  }

  counts  <- counts[, keep_lib, drop = FALSE]
  coldata <- coldata[ keep_lib, , drop = FALSE]
  libsizes <- libsizes[keep_lib]

  ## -------------------- DESIGN MATRIX -----------------------
  use_batch <- ("batch" %in% colnames(coldata)) &&
               (length(unique(coldata$batch)) > 1) &&
               (nrow(table(coldata$batch, coldata$condition)) >
                  min(length(unique(coldata$batch)),
                      length(levels(coldata$condition))))

  if (use_batch) {
    message("  Using design: ~ condition + batch")
    design <- model.matrix(~ condition + batch, data = coldata)
  } else {
    message("  Using design: ~ condition")
    design <- model.matrix(~ condition, data = coldata)
  }

  ## -------------------- FEATURE FILTERING -------------------
  # keep peaks with non-zero signal in at least 2 samples
  nz_per_feature <- rowSums(counts > 0, na.rm = TRUE)
  keep_feat <- nz_per_feature >= 2

  message("  Features with signal in >=2 samples: ",
          sum(keep_feat), " of ", length(nz_per_feature))

  if (sum(keep_feat) < 20L) {
    message("  ! Too few features with signal; skipping")
    next
  }

  counts_filt <- counts[keep_feat, , drop = FALSE]

  ## -------------------- DGE WITH PSEUDO-COUNTS --------------
  # Your values are tiny fractional accessibility; scale to pseudo-counts
  scale_factor <- 1e6
  counts_scaled <- round(counts_filt * scale_factor)

  y <- DGEList(counts_scaled)
  y <- calcNormFactors(y)
  y <- estimateDisp(y, design)
  fit <- glmQLFit(y, design)

  if (!("conditionAggressive" %in% colnames(coef(fit)))) {
    message("  ! Coefficient 'conditionAggressive' not found; skipping")
    next
  }

  qlf <- glmQLFTest(fit, coef = "conditionAggressive")
  tab <- topTags(qlf, n = Inf)$table
  tab$gene <- rownames(tab)

  out <- paste0(base, "_edgeR_DGE.csv")
  write.csv(tab, out, row.names = FALSE)
  message("  Saved: ", out)
}
