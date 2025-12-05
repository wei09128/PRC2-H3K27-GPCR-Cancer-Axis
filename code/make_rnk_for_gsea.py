#!/usr/bin/env python3
import pandas as pd
import numpy as np

def make_rnk(in_csv, out_rnk):
    print(f"Reading {in_csv}")
    df = pd.read_csv(in_csv)

    # Keep rows with proper values
    df = df.dropna(subset=["gene", "logFC", "PValue"])

    # Signed ranking statistic: + if higher in Aggressive, - if higher in less-Aggressive
    df["stat"] = np.sign(df["logFC"]) * -np.log10(df["PValue"] + 1e-300)

    # Drop genes with stat == 0 just to be safe
    df = df[df["stat"] != 0].copy()

    # Sort by stat descending
    df = df.sort_values("stat", ascending=False)

    # Build .rnk: two columns: gene, stat (no header)
    rnk = df[["gene", "stat"]]
    rnk.to_csv(out_rnk, sep="\t", index=False, header=False)

    print(f"  Wrote {len(rnk)} genes → {out_rnk}")

if __name__ == "__main__":
    # scChi
    make_rnk(
        "/mnt/f/H3K27/Data/scChi_pseudo/scChi_pseudobulk_edgeR_DGE.csv",
        "/mnt/f/H3K27/Data/scChi_pseudo/scChi_pseudobulk_forGSEA.rnk"
    )

    # scATAC
    make_rnk(
        "/mnt/f/H3K27/Data/scATAC_pseudo/scATAC_BC_pseudobulk_edgeR_DGE.csv",
        "/mnt/f/H3K27/Data/scATAC_pseudo/scATAC_pseudobulk_forGSEA.rnk"
    )
