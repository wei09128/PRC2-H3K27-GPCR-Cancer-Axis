import anndata as ad

in_path  = '/mnt/f/H3K27/Data/scATAC/1/scATAC_1000k.h5ad'
out_path = '/mnt/f/H3K27/Data/scATAC/1/scATAC_1000k_fixed.h5ad'
# in_path  = '/mnt/f/H3K27/Data/scChi/1/scChi_1000k.h5ad'
# out_path = '/mnt/f/H3K27/Data/scChi/1/scChi_1000k_fixed.h5ad'

adata = ad.read_h5ad(in_path)

def clean_sample_id_scATAC(s: str) -> str:
    s = str(s)
    # A_ / B_ cases: "A_A10_Ovarian_IVB_Unknown_ATAC" -> "A10"
    if s.startswith("A_") or s.startswith("B_"):
        parts = s.split("_")
        if len(parts) > 1:
            return parts[1]
        return s
    # Cell line cases: "CellLine_HCC1143_Breast_TNBC_Sensitive_ATAC" -> "HCC1143"
    if s.startswith("CellLine_"):
        parts = s.split("_")
        if len(parts) > 1:
            return parts[1]
        return s
    # Otherwise, leave unchanged
    return s

print("Sample_ID before (first 10):")
print(adata.obs["Sample_ID"].astype(str).unique()[:10])

adata.obs["Sample_ID"] = adata.obs["Sample_ID"].astype(str).map(clean_sample_id_scATAC)

print("Sample_ID after (first 10):")
print(adata.obs["Sample_ID"].astype(str).unique()[:10])

adata.write_h5ad(out_path)
print(f"Saved → {out_path}")
