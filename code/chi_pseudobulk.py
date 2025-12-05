#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import scanpy as sc

IN_H5AD    = "/mnt/f/H3K27/Data/scChi/scChi_gene_QC.h5ad"
OUT_DIR    = "/mnt/f/H3K27/Data/scChi_pseudo/"
GROUP_COL  = "Response_Status"   # ['less-Aggressive','Persister','Resistant','Aggressive']
SAMPLE_COL = "Sample_ID"
MARK_COL   = "Chi"
MARK_KEEP  = "H3K27me3"          # keep H3K27me3 mark

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading: {IN_H5AD}")
adata = sc.read_h5ad(IN_H5AD)

# --- 1) Filter to the ChIP mark you want ---
if MARK_COL not in adata.obs.columns:
    raise ValueError(f"Column '{MARK_COL}' not found in adata.obs")

print("Available Chi levels:", adata.obs[MARK_COL].unique().tolist())

mask_mark = adata.obs[MARK_COL] == MARK_KEEP
adata = adata[mask_mark, :].copy()
print(f"After filtering to {MARK_KEEP}: {adata.n_obs} cells")

# --- 2) (Optional) sanity check on Response_Status ---
print("Response_Status levels in filtered data:",
      adata.obs[GROUP_COL].unique().tolist())

# --- 3) Build pseudobulk: Sample_ID × gene ---
samples  = adata.obs[SAMPLE_COL].unique().tolist()
features = adata.var_names.tolist()

pb_matrix    = []
pb_meta_rows = []

for sid in samples:
    idx = adata.obs[SAMPLE_COL] == sid
    sub = adata[idx, :]

    # sum across cells (sparse or dense)
    summed = np.asarray(sub.X.sum(axis=0)).ravel()
    pb_matrix.append(summed)

    meta_row = {
        "Sample_ID":      sid,
        "Response_Status": sub.obs[GROUP_COL].iloc[0],
        "Cancer_Type":     sub.obs.get("Cancer_Type",
                                       pd.Series(["NA"])).iloc[0],
        "Batch_ID":        sub.obs.get("Batch_ID",
                                       pd.Series(["NA"])).iloc[0],
        "Chi":             sub.obs.get("Chi",
                                       pd.Series(["NA"])).iloc[0],
    }
    pb_meta_rows.append(meta_row)

pb_matrix = np.vstack(pb_matrix)
pb_meta   = pd.DataFrame(pb_meta_rows).set_index("Sample_ID")

print("Pseudobulk matrix shape:", pb_matrix.shape)
print(pb_meta.head())

# --- 4) Save counts + metadata ---
counts_df = pd.DataFrame(pb_matrix,
                         index=pb_meta.index,
                         columns=features)

counts_df.to_csv(os.path.join(OUT_DIR, "scChi_pseudobulk_counts.csv"))
pb_meta.to_csv(os.path.join(OUT_DIR, "scChi_pseudobulk_metadata.csv"))

print("Saved:")
print(" - scChi_pseudobulk_counts.csv")
print(" - scChi_pseudobulk_metadata.csv")
