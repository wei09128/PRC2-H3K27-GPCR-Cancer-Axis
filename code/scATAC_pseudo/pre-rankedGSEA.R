#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fgsea)
  library(data.table)
})

## --------------------------------------------------
## 1) Load pathways (GMT)
## --------------------------------------------------

# EDIT THIS: path to your gene-set GMT file
# e.g. the same one used for scRNA GSEA (H3K27, Hallmarks, STRENGTH, etc.)
pathways_gmt <- "/mnt/f/H3K27/Data/CCC/c2.all.v2023.1.Hs.symbols.gmt"  ### EDIT THIS ###
if (!file.exists(pathways_gmt)) {
  stop("GMT file not found: ", pathways_gmt)
}

pathways <- gmtPathways(pathways_gmt)
cat("Loaded", length(pathways), "pathways from GMT\n")

## --------------------------------------------------
## 2) Load ranked stats (.rnk) for scATAC
## --------------------------------------------------

rnk_file_atac <- "/mnt/f/H3K27/Data/scATAC_pseudo/scATAC_pseudobulk_forGSEA.rnk"  ### EDIT THIS ###
if (!file.exists(rnk_file_atac)) {
  stop("RNK file not found: ", rnk_file_atac)
}

rnk_atac <- fread(rnk_file_atac, header = FALSE)
if (ncol(rnk_atac) < 2) {
  stop("RNK file should have two columns: gene, score")
}

stats_atac <- setNames(rnk_atac$V2, rnk_atac$V1)

# clean: drop NA, deduplicate by max |stat|
stats_atac <- stats_atac[!is.na(stats_atac)]
stats_atac <- tapply(stats_atac, names(stats_atac), function(x) x[which.max(abs(x))])
stats_atac <- sort(unlist(stats_atac), decreasing = TRUE)

cat("Length of stats_atac after cleaning:", length(stats_atac), "\n")

## --------------------------------------------------
## 3) Quick overlap sanity check
## --------------------------------------------------

overlaps <- sapply(pathways, function(g) sum(g %in% names(stats_atac)))
cat("Pathways with >= 5 overlapping genes: ", sum(overlaps >= 5), "of", length(pathways), "\n")

if (sum(overlaps >= 5) == 0) {
  warning("No pathways have at least 5 overlapping genes with stats_atac.\n",
          "Check that gene IDs in GMT (symbols vs ENSG...) match the RNK.")
}

## --------------------------------------------------
## 4) Run pre-ranked GSEA with fgseaMultilevel
## --------------------------------------------------

fg_atac <- fgseaMultilevel(
  pathways = pathways,
  stats    = stats_atac,
  minSize  = 10,
  maxSize  = 500
)

# sort by adjusted p-value
fg_atac <- fg_atac[order(fg_atac$padj), ]
cat("Top 5 pathways:\n")
print(head(fg_atac, 5))

# save results
out_csv_atac <- "scATAC_pseudobulk_fgsea_results.csv"
fwrite(fg_atac, out_csv_atac)
cat("Saved ATAC fgsea results to:", out_csv_atac, "\n")
