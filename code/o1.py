import pandas as pd

df = pd.read_csv("/mnt/f/H3K27/Data/scATAC_pseudo/scATAC_pseudobulk_edgeR_DGE.csv")
sig = df[(df["FDR"] < 0.05) & (df["logFC"].abs() > 0.5)]
sig[["gene","logFC","FDR"]].head(20)
