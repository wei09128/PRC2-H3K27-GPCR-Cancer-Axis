import pandas as pd
import matplotlib.pyplot as plt
import os

# IN = "/mnt/f/H3K27/Data/scATAC_pseudo/scATAC_pseudobulk_edgeR_DGE.csv"
# OUT = "/mnt/f/H3K27/Data/scATAC_pseudo/scATAC_top_genes_barplot.png"
IN = "/mnt/f/H3K27/Data/scChi_pseudo/scChi_pseudobulk_edgeR_DGE.csv"
OUT = "/mnt/f/H3K27/Data/scChi_pseudo/scChi_top_genes_barplot.png"

df = pd.read_csv(IN)

# 1) keep only scATAC rows if you ever combine multiple CTs/modalities
# df = df[df["cell_type"] == "scATAC"].copy()

# 2) decide what to plot: e.g. top 15 by FDR
top = df.sort_values("FDR").head(15)

# 3) plot horizontal barplot
plt.figure(figsize=(5, 6))
plt.barh(top["gene"], top["logFC"])
plt.axvline(0, linewidth=0.8)

plt.xlabel("log2 fold change (Aggressive vs less-aggressive)")
# plt.ylabel("Gene (ATAC pseudobulk)")
# plt.title("Top scATAC DGE genes (pseudobulk)")

plt.ylabel("Gene (Chi pseudobulk)")
plt.title("Top scChi DGE genes (pseudobulk)")

plt.tight_layout()
plt.savefig(OUT, dpi=300)
plt.close()

print("Saved barplot to:", OUT)
