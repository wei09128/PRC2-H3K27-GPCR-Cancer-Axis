import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the DPI for saving figures
DPI_SETTING = 300 

# --- 1. Load the h5ad file ---
try:
    # IMPORTANT: Update 'your_data.h5ad' with the actual file path/name
    adata = sc.read_h5ad('/mnt/f/H3K27/Data/scRNA/scRNA_final.h5ad') 
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: H5AD file not found. Please check the file path.")
    exit()

if 'raw' not in adata.layers:
    print("Warning: Assuming data in adata.X is raw counts for QC.")

# # Filter based on the recommended thresholds
# # 1. Minimum Gene Count (remove debris)
# adata = adata[adata.obs.n_genes_by_counts >= 200, :]
# # 2. Maximum Gene Count (remove probable doublets)
# adata = adata[adata.obs.n_genes_by_counts <= 3000, :]
# # 3. Maximum Mitochondrial Percentage (remove stressed/dead cells)
# adata = adata[adata.obs.pct_counts_mt <= 10, :]

# --- 2. Calculate QC Metrics ---
# Adjust to 'mt-' if working with mouse data.
adata.var['mt'] = adata.var_names.str.startswith('MT-') 
sc.pp.calculate_qc_metrics(
    adata, 
    qc_vars=['mt'], 
    percent_top=None, 
    log1p=False, 
    inplace=True
)

print("\nQC Metrics calculated and stored in adata.obs:")
print(adata.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt']].head())


# --- 3. Visualize and Save QC Metrics (Violin Plots) ---

plt.rcParams["figure.figsize"] = (6, 4) 
print("\nSaving Violin Plots...")

# UMI Counts Plot
sc.pl.violin(
    adata, 
    keys='total_counts', 
    log=True, 
    show=False
)
plt.title('UMI Counts (total_counts) Distribution')
plt.tight_layout()
plt.savefig('/mnt/f/H3K27/Data/scRNA/QC_Violin_UMI_Counts.png', dpi=DPI_SETTING)
plt.close() # Close the figure to free memory

# Gene Counts Plot
sc.pl.violin(
    adata, 
    keys='n_genes_by_counts', 
    show=False
)
plt.title('Gene Counts (n_genes_by_counts) Distribution')
plt.tight_layout()
plt.savefig('/mnt/f/H3K27/Data/scRNA/QC_Violin_Gene_Counts.png', dpi=DPI_SETTING)
plt.close()

# Mitochondrial Percentage Plot
sc.pl.violin(
    adata, 
    keys='pct_counts_mt', 
    show=False
)
plt.title('Mitochondrial Percentage (pct_counts_mt) Distribution')
plt.tight_layout()
plt.savefig('/mnt/f/H3K27/Data/scRNA/QC_Violin_Mitochondrial_Percentage.png', dpi=DPI_SETTING)
plt.close()


# --- 4. Visualize and Save Scatter Plot ---
print("Saving Scatter Plot...")

# Scatter Plot (UMI vs. Genes, colored by MT%)
sc.pl.scatter(
    adata, 
    x='total_counts', 
    y='n_genes_by_counts', 
    color='pct_counts_mt',
    show=False
)
plt.title('UMI vs. Gene Counts, Colored by MT Percentage')
plt.tight_layout()
plt.savefig('/mnt/f/H3K27/Data/scRNA/QC_Scatter_UMI_Gene_MT.png', dpi=DPI_SETTING)
plt.close()

print(f"All 4 QC plots saved successfully in the current directory with {DPI_SETTING} DPI.")