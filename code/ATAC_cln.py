#!/usr/bin/env python3
import scanpy as sc
import numpy as np
import scipy.sparse as sp

IN_H5AD  = "/mnt/f/H3K27/Data/scATAC/scATAC_peaks_QC_float.h5ad"
OUT_H5AD = "/mnt/f/H3K27/Data/scATAC/scATAC_peaks_QC_BC_float64_clean.h5ad"

print(f"Loading {IN_H5AD}")
adata = sc.read_h5ad(IN_H5AD)

# 1) Restrict to BC only
if "Cancer_Type" in adata.obs.columns:
    bc_mask = adata.obs["Cancer_Type"] == "BC"
    adata = adata[bc_mask, :].copy()
    print(f"After restricting to BC only: {adata.n_obs} cells")
    print("Cancer types now:", adata.obs["Cancer_Type"].unique())
else:
    print("WARNING: no 'Cancer_Type' in .obs; not subsetting")

# 2) Drop raw and all layers (these often keep int matrices!)
adata.layers.clear()
adata.raw = None
print("Cleared layers and raw.")

# 3) Force X to be sparse CSR with float64
if sp.issparse(adata.X):
    adata.X = adata.X.astype(np.float64)
    adata.X = adata.X.tocsr()  # explicit CSR double
else:
    adata.X = np.asarray(adata.X, dtype=np.float64)

print("X dtype:", adata.X.dtype)

# 4) (Optional but safe) drop unnecessary stuff that might confuse R
for attr in ["uns", "obsm", "varm"]:
    if hasattr(adata, attr):
        # keep, but you can uncomment next line if you want it ultra-minimal
        # setattr(adata, attr, {})
        pass

adata.write_h5ad(OUT_H5AD, compression="gzip")
print("Written:", OUT_H5AD)
