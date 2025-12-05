import anndata as ad

in_path  = "/mnt/f/H3K27/Data/scChi/1/scChi_1000k_fixed.h5ad"
out_path = "/mnt/f/H3K27/Data/scChi/1/scChi_1000k_fixed1.h5ad"

adata = ad.read_h5ad(in_path)

print("Sample_ID before:")
print(adata.obs["Sample_ID"].astype(str).unique())

def clean_sample_id_scChi(s: str) -> str:
    s = str(s)
    if s.startswith("C_C1K27"):
        return "C1K27"
    if s.startswith("C_C1K4"):
        return "C1K4"
    # everything else already OK (MM468-... etc.)
    return s

adata.obs["Sample_ID"] = adata.obs["Sample_ID"].astype(str).map(clean_sample_id_scChi)

print("\nSample_ID after:")
print(adata.obs["Sample_ID"].astype(str).unique())

adata.write_h5ad(out_path)
print(f"\nSaved → {out_path}")
