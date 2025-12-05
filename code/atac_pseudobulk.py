#!/usr/bin/env python3 
import os
import numpy as np
import pandas as pd
import scanpy as sc

IN_H5AD    = "/mnt/f/H3K27/Data/scATAC/scATAC_gene_QC.h5ad"
OUT_DIR    = "/mnt/f/H3K27/Data/scATAC_pseudo/"   # optional: BC-specific output
GROUP_COL  = "Response_Status"   # Aggressive vs less-Aggressive
SAMPLE_COL = "Sample_ID"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading: {IN_H5AD}")
adata = sc.read_h5ad(IN_H5AD)

# --- 2.0 Restrict to Breast Cancer only ---
bc_mask = adata.obs["Cancer_Type"] == "BC"
adata = adata[bc_mask, :].copy()
print(f"After restricting to BC only: {adata.n_obs} cells")
print("Cancer_Types present:", adata.obs["Cancer_Type"].unique().tolist())

# --- 2.1 Filter to the groups you care about (Aggressive vs less-Aggressive) ---
keep_groups = ["Aggressive", "less-Aggressive"]
mask = adata.obs[GROUP_COL].isin(keep_groups)
adata = adata[mask, :].copy()
print(f"Filtered to {adata.n_obs} BC cells in groups {keep_groups}")

# --- 2.2 Build pseudobulk counts: Sample_ID × peak ---
samples  = adata.obs[SAMPLE_COL].unique().tolist()
features = adata.var_names.tolist()

pb_matrix    = []
pb_meta_rows = []

for sid in samples:
    idx = adata.obs[SAMPLE_COL] == sid
    sub = adata[idx, :]

    # sum across cells (axis=0); works for sparse or dense
    summed = np.asarray(sub.X.sum(axis=0)).ravel()
    pb_matrix.append(summed)

    # representative metadata row for this sample
    meta_row = {
        "Sample_ID": sid,
        "Response_Status": sub.obs[GROUP_COL].iloc[0],
        "Cancer_Type": sub.obs.get("Cancer_Type", pd.Series(["NA"])).iloc[0],
        "Batch_ID": sub.obs.get("Batch_ID", pd.Series(["NA"])).iloc[0],
    }
    pb_meta_rows.append(meta_row)

pb_matrix = np.vstack(pb_matrix)  # shape: [n_samples, n_features]
pb_meta   = pd.DataFrame(pb_meta_rows).set_index("Sample_ID")

print("Pseudobulk matrix shape:", pb_matrix.shape)
print("Pseudobulk metadata:")
print(pb_meta.head())

# --- 2.3 Save outputs for downstream DE (edgeR/DESeq2, etc.) ---

# (a) counts matrix: rows=Sample_ID, columns=features (peaks/genes)
counts_df = pd.DataFrame(pb_matrix,
                         index=pb_meta.index,
                         columns=features)
counts_df.to_csv(os.path.join(OUT_DIR, "scATAC_BC_pseudobulk_counts.csv"))

# (b) sample metadata for design matrix
pb_meta.to_csv(os.path.join(OUT_DIR, "scATAC_BC_pseudobulk_metadata.csv"))

print("Saved:")
print(" - scATAC_BC_pseudobulk_counts.csv")
print(" - scATAC_BC_pseudobulk_metadata.csv")
