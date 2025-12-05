import scanpy as sc
import pandas as pd

# Paths
# h5ad_path = '/mnt/f/H3K27/Data/scRNA/scRNA_save2.h5ad'
# metadata_path = '/mnt/f/H3K27/Data/metadata.csv'
# output_path = '/mnt/f/H3K27/Data/scRNA/scRNA_final.h5ad'
# join_string='Condition' # The column to use for joining (e.g., 'Sample_ID' or 'Condition')

metadata_path = '/mnt/f/H3K27/Data/metadata_ATAC_Chi.csv'
# h5ad_path = '/mnt/f/H3K27/Data/scATAC/1/scATAC_1000k_fixed.h5ad'
# output_path = '/mnt/f/H3K27/Data/scATAC/1/scATAC_1000k_fixed2.h5ad'
h5ad_path = '/mnt/f/H3K27/Data/scChi/1/scChi_1000k_fixed2_rawX.h5ad'
output_path = '/mnt/f/H3K27/Data/scChi/1/scChi_1000k_fixed2_rawX.h5ad'

# h5ad_path = '/mnt/f/H3K27/Data/scChi/scChi_accessibility.h5ad'
# output_path = '/mnt/f/H3K27/Data/scChi/scChi_accessibility.h5ad'
# --- Define Join Parameters ---
join_string='Sample_ID' # The column to use for joining (e.g., 'Sample_ID' or 'Condition')
all_colums = ['Batch_ID',join_string, 'Cancer_Type', 'Cancer_Status', 'Response_Status', 'Condition','Chi', ]

# We use this list to select the columns from the metadata CSV before joining.

# Load AnnData
adata = sc.read_h5ad(h5ad_path)
print(f"Loaded AnnData with {adata.n_obs} cells.")

# --- Load metadata robustly ---
meta = pd.read_csv(metadata_path, sep=',')
meta.columns = meta.columns.str.strip()
print("✅ Loaded metadata columns:", meta.columns.tolist())

# If only one big column detected, retry with whitespace separator
if len(meta.columns) == 1:
    meta = pd.read_csv(metadata_path, sep=r'\s+')
    meta.columns = meta.columns.str.strip()
    print("⚙️ Retried with whitespace separator → columns:", meta.columns.tolist())

# Check overlap using the defined join key (join_string)
if join_string not in meta.columns:
    raise ValueError(f"❌ Join key '{join_string}' not found in metadata. Found: {meta.columns.tolist()}")

if join_string not in adata.obs.columns:
    # If the desired join column isn't in adata.obs, the join will fail.
    # We must raise an error OR default to a known column like 'Sample_ID'
    raise ValueError(f"❌ Join key '{join_string}' not found in adata.obs. Please check key or use 'Sample_ID'.")

common_ids = set(adata.obs[join_string]).intersection(set(meta[join_string]))
print(f"Matched {len(common_ids)} of {len(adata.obs[join_string].unique())} unique IDs on column '{join_string}'.")

# --- Prepare metadata to join ---
meta_to_join = meta[all_colums].drop_duplicates(subset=join_string)
meta_to_join = meta_to_join.set_index(join_string)

# --- Inspect column overlap --- 
existing_cols = set(adata.obs.columns)
meta_cols = set(meta_to_join.columns)
missing_cols = meta_cols - existing_cols

print(f"🧩 Columns in metadata not yet in adata.obs: {sorted(missing_cols)}")
print(f"⚠️ Overlapping columns (already exist in adata.obs): {sorted(meta_cols & existing_cols)}")

# --- Overwrite / update columns: direct join (no .update) ---
# Drop any old versions of these columns, then join the metadata
adata.obs = (
    adata.obs
    .drop(columns=list(meta_to_join.columns), errors='ignore')
    .join(meta_to_join, on=join_string, how='left')
)

# Optionally cast new metadata columns to category
for col in meta_to_join.columns:
    if col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype('category')

# --- Verify ---
cols_to_show = [c for c in all_colums if c in adata.obs.columns]
print("\n--- Head of updated adata.obs ---")
print(adata.obs[cols_to_show].head())

# --- SAVE UPDATED H5AD ---
adata.write_h5ad(output_path)
print(f"\n✅ Updated AnnData saved → {output_path}")
