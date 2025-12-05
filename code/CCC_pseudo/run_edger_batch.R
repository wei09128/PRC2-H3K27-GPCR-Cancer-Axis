# run_edger_batch.R — edgeR on all pseudobulk counts/coldata in this folder
# Writes: pseudobulk_<CT>_edgeR_DGE.csv

# ----- ensure and register a user-writable library -----
userlib <- Sys.getenv("R_LIBS_USER")
if (userlib == "") {
  userlib <- file.path(Sys.getenv("HOME"), "R",
                       paste0(getRversion()$major,".",getRversion()$minor),
                       "library")
  Sys.setenv(R_LIBS_USER = userlib)
}
dir.create(userlib, recursive = TRUE, showWarnings = FALSE)
.libPaths(unique(c(userlib, .libPaths())))   # <-- make userlib visible NOW

# ----- install/load packages into user library if needed -----
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org", lib = userlib)
  .libPaths(unique(c(userlib, .libPaths()))) # refresh paths
}
if (!requireNamespace("edgeR", quietly = TRUE) || !requireNamespace("limma", quietly = TRUE)) {
  # install into userlib
  BiocManager::install(c("edgeR","limma"), ask = FALSE, update = FALSE, lib = userlib)
  .libPaths(unique(c(userlib, .libPaths()))) # refresh paths
}

suppressPackageStartupMessages({
  library(edgeR, lib.loc = userlib)
  library(limma,  lib.loc = userlib)
})

# ----- process all pseudobulk pairs -----
dir <- "."
cfs <- list.files(dir, pattern = "^pseudobulk_.*_counts\\.csv$", full.names = TRUE)

for (cf in cfs) {
  base <- sub("_counts\\.csv$", "", cf)
  colf <- paste0(base, "_coldata.csv")

  message("\n=== ", basename(base), " ===")
  counts  <- read.csv(cf,  row.names = 1, check.names = FALSE)
  coldata <- read.csv(colf, row.names = 1, check.names = FALSE)

  # align
  common <- intersect(colnames(counts), rownames(coldata))
  if (length(common) < 4L) { message("  ! Skipping (too few samples)"); next }
  counts  <- counts[,  common, drop = FALSE]
  coldata <- coldata[ common, , drop = FALSE]

  # factors
  if (!("condition" %in% colnames(coldata))) { message("  ! No 'condition' column; skipping"); next }
  coldata$condition <- factor(coldata$condition, levels = c("Control","Aggressive"))

  # optional batch (only if useful and not collinear)
  use_batch <- ("batch" %in% colnames(coldata)) &&
               (length(unique(coldata$batch)) > 1) &&
               (nrow(table(coldata$batch, coldata$condition)) >
                   min(length(unique(coldata$batch)),
                       length(levels(coldata$condition))))
  design <- if (use_batch) model.matrix(~ condition + batch, data = coldata)
            else            model.matrix(~ condition,        data = coldata)

  # edgeR QLF
  y <- DGEList(counts)
  keep <- filterByExpr(y, design)
  y <- y[keep,, keep.lib.sizes = FALSE]
  if (ncol(y) < 4L || nrow(y) < 50L) { message("  ! Too small after filter; skipping"); next }

  y <- calcNormFactors(y)
  y <- estimateDisp(y, design)
  fit <- glmQLFit(y, design)

  qlf <- glmQLFTest(fit, coef = "conditionAggressive")
  tab <- topTags(qlf, n = Inf)$table
  tab$gene <- rownames(tab)

  out <- paste0(base, "_edgeR_DGE.csv")
  write.csv(tab, out, row.names = FALSE)
  message("  Saved: ", out)
}
