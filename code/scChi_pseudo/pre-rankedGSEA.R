#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fgsea)
  library(data.table)
  library(ggplot2)
})

## -------- paths --------
gmt_path <- "/mnt/f/H3K27/Data/CCC/c2.all.v2023.1.Hs.symbols.gmt"
rnk_path <- "scChi_pseudobulk_forGSEA.rnk"   # your .rnk file in this folder
out_csv  <- "scChi_pseudobulk_fgsea_results.csv"
out_png  <- "scChi_pseudobulk_GSEA_H3K27_barplot.png"

cat("Loading pathways from:\n ", gmt_path, "\n")
pathways_all <- gmtPathways(gmt_path)
cat("Loaded", length(pathways_all), "pathways\n")

## -------- load ranked stats --------
cat("Loading ranked stats from:\n ", rnk_path, "\n")
rnk <- fread(rnk_path, header = FALSE)
if (nrow(rnk) == 0) {
  stop("Rank file has 0 rows – check scChi_pseudobulk_forGSEA.rnk")
}

stats <- setNames(rnk$V2, rnk$V1)
stats <- stats[!is.na(stats)]
stats <- sort(stats, decreasing = TRUE)
cat("Number of genes in stats:", length(stats), "\n")

## -------- run fgsea --------
cat("Running fgseaMultilevel ...\n")
fg_res <- fgseaMultilevel(
  pathways = pathways_all,
  stats    = stats,
  minSize  = 10,
  maxSize  = 500
)

## save full table
fg_res <- fg_res[order(padj)]
fwrite(fg_res, out_csv)
cat("Saved full fgsea table to:", out_csv, "\n")

## -------- focus on H3K27 / PRC2 gene sets --------
h3k27_idx <- grepl("H3K27|K27ME3|PRC2|EZH2", fg_res$pathway, ignore.case = TRUE)
h3k27 <- fg_res[h3k27_idx]

if (nrow(h3k27) == 0) {
  cat("No H3K27/PRC2-related pathways found – no barplot written.\n")
  quit(save = "no")
}

## take top 10 by padj (or all if <10)
h3k27 <- h3k27[order(padj)][1:min(10, nrow(h3k27))]

cat("Top H3K27-related sets:\n")
print(h3k27[, .(pathway, pval, padj, ES, NES, size)])

## order for plotting (lowest padj at top)
h3k27[, pathway_clean := factor(pathway, levels = rev(pathway))]

p <- ggplot(h3k27, aes(x = NES, y = pathway_clean, fill = NES)) +
  geom_col() +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  scale_fill_gradient2(
    low  = "#2166ac",
    mid  = "white",
    high = "#b2182b",
    midpoint = 0
  ) +
  xlab("Normalized Enrichment Score (NES)") +
  ylab("H3K27 / PRC2 gene sets (scChi pseudobulk)") +
  theme_minimal(base_size = 10) +
  theme(
    legend.position   = "right",
    axis.text.y       = element_text(size = 7),
    axis.text.x       = element_text(size = 8),
    plot.title        = element_text(size = 11, face = "bold")
  ) +
  ggtitle("Pre-ranked GSEA of scChi pseudobulk (Aggressive vs less-aggressive)")

ggsave(
  filename = out_png,
  plot     = p,
  width    = 4,
  height   = 3,
  dpi      = 300
)

cat("Saved H3K27 barplot to:", out_png, "\n")
