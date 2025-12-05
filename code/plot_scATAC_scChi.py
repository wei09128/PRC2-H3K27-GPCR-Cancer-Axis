import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load table
# df = pd.read_csv("/mnt/f/H3K27/Data/scATAC_pseudo/scATAC_pseudobulk_fgsea_results.csv")
df = pd.read_csv("/mnt/f/H3K27/Data/scChi_pseudo/scChi_pseudobulk_fgsea_results.csv")

# keep top significant pathways
df = df.sort_values("padj")
top = df.head(15).copy()   # change 15 -> 10/20 as you like

# sort for plotting (so bars go nicely)
top = top.sort_values("NES")

plt.figure(figsize=(6,5))
bars = plt.barh(top["pathway"], top["NES"])

# optional: fade by significance
# (smaller padj = darker)
padj = top["padj"].clip(lower=1e-300)
alphas = 1 - (np.log10(padj) / np.log10(padj.max()))
alphas = np.clip(alphas, 0.2, 1.0)
for b, a in zip(bars, alphas):
    b.set_alpha(a)

plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Normalized Enrichment Score (NES)")
# plt.title("scATAC GSEA: top pathways by FDR")
plt.title("scChi GSEA: top pathways by FDR")

plt.tight_layout()
# out = "/mnt/f/H3K27/Data/scATAC/atac_gsea_topNES.png"
out = "/mnt/f/H3K27/Data/scChi/chi_gsea_topNES.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print("Saved:", out)
plt.show()
