#!/usr/bin/env python3
"""
Batch Doublet Filtering for scRNA-seq
- Uses Scrublet or MAD-based filtering
- Guarantees integer raw counts for Scrublet
- Filters poor-quality cells before Scrublet
- Recomputes PCA/neighbors after filtering
"""

import os
import scanpy as sc
import scrublet as scr
import anndata as ad
import numpy as np
import pandas as pd
import warnings
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# --- HELPER FUNCTIONS ---
# =============================================================================

def get_raw_matrix(adata: ad.AnnData):
    """
    Safely find raw integer counts matrix for Scrublet.
    Priority: layers['raw_counts'] > .raw.X > .X
    """
    if 'raw_counts' in adata.layers:
        print("  ✓ Using layers['raw_counts'] as raw input.")
        matrix = adata.layers['raw_counts']
    elif adata.raw is not None:
        print("  ✓ Using .raw.X as raw input.")
        matrix = adata.raw.X
    else:
        print("  ⚠ Using .X (verify it's raw counts).")
        matrix = adata.X

    # Convert sparse to dense if needed for Scrublet
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    # --- Validation ---
    is_integer = np.allclose(matrix[:100, :100], np.round(matrix[:100, :100]))
    max_val = np.max(matrix)
    if not is_integer or max_val < 20:
        raise ValueError("❌ Scrublet requires integer UMI counts. Your input seems normalized or log-transformed.")

    return matrix


# =============================================================================
# --- SCRUBLET DOUBLETS ---
# =============================================================================

def filter_doublets_scrublet(
    adata: ad.AnnData,
    expected_doublet_rate: float = 0.06,
    min_counts: int = 2,
    min_cells: int = 3,
    n_prin_comps: int = 30
) -> ad.AnnData:
    """
    Filters doublets using Scrublet on verified raw counts.
    """
    print(f"\n[Scrublet] Original cells: {adata.n_obs}")

    # --- 1. Basic pre-filtering for quality ---
    sc.pp.filter_cells(adata, min_counts=250)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  ✓ After QC filter: {adata.n_obs} cells, {adata.n_vars} genes")

    # --- 2. Get validated raw count matrix ---
    scrub_matrix = get_raw_matrix(adata)

    # --- 3. Run Scrublet ---
    print("  > Running Scrublet...")
    scrub = scr.Scrublet(scrub_matrix, expected_doublet_rate=expected_doublet_rate, random_state=42)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=min_counts,
        min_cells=min_cells,
        n_prin_comps=n_prin_comps
    )

    adata.obs['doublet_score'] = doublet_scores
    adata.obs['predicted_doublet'] = predicted_doublets

    # --- 4. Plot histogram for QC ---
    # scrub.plot_histogram() returns (Figure, Axes), so we unpack the figure object.
    fig, ax = scrub.plot_histogram()  # <--- CORRECTED LINE: Unpack the tuple
    plt.title("Scrublet Doublet Score Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "doublet_score_hist.png"), dpi=150)
    plt.close(fig) # <--- 'fig' is now correctly the Figure object

    # --- 5. Filter singlets ---
    singlet_mask = ~predicted_doublets
    adata_singlets = adata[singlet_mask].copy()

    removed = adata.n_obs - adata_singlets.n_obs
    print(f"  ✓ Doublets removed: {removed} ({removed / adata.n_obs * 100:.2f}%)")
    print(f"  ✓ Remaining: {adata_singlets.n_obs}")

    # --- 6. Recompute PCA/neighbors for downstream analysis ---
    #sc.pp.normalize_total(adata_singlets, target_sum=1e4)
    #sc.pp.log1p(adata_singlets)
    #sc.pp.highly_variable_genes(adata_singlets, n_top_genes=2000)
    #adata_singlets = adata_singlets[:, adata_singlets.var['highly_variable']]
    #sc.pp.scale(adata_singlets, max_value=10)
    #sc.tl.pca(adata_singlets, svd_solver='arpack')
    #sc.pp.neighbors(adata_singlets, n_pcs=30)
    #sc.tl.umap(adata_singlets)
    #print("  ✓ PCA and neighbors recomputed on filtered data.")
    print("  ✓ QC and doublet removal complete. Output matrix contains filtered raw counts.")

    return adata_singlets


# =============================================================================
# --- SIMPLE MAD DOUBLETS ---
# =============================================================================

def filter_doublets_simple(
    adata: ad.AnnData,
    mad_threshold: float = 5.0,
    use_n_genes: bool = True
) -> ad.AnnData:
    """
    Simple MAD-based filtering for doublets.
    """
    print(f"\n[MAD] Original cells: {adata.n_obs}")
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    singlet_mask = np.ones(adata.n_obs, dtype=bool)

    counts = adata.obs['total_counts'].values
    median_counts = np.median(counts)
    mad_counts = median_abs_deviation(counts)
    upper_counts = median_counts + mad_threshold * mad_counts
    singlet_mask &= counts < upper_counts
    print(f"  > total_counts threshold: {upper_counts:.0f}")

    if use_n_genes:
        n_genes = adata.obs['n_genes_by_counts'].values
        median_genes = np.median(n_genes)
        mad_genes = median_abs_deviation(n_genes)
        upper_genes = median_genes + mad_threshold * mad_genes
        singlet_mask &= n_genes < upper_genes
        print(f"  > n_genes threshold: {upper_genes:.0f}")

    adata.obs['predicted_doublet_mad'] = ~singlet_mask
    adata_singlets = adata[singlet_mask].copy()

    removed = adata.n_obs - adata_singlets.n_obs
    print(f"  ✓ Doublets removed: {removed} ({removed / adata.n_obs * 100:.2f}%)")
    print(f"  ✓ Remaining: {adata_singlets.n_obs}")
    return adata_singlets


# =============================================================================
# --- BATCH DRIVER ---
# =============================================================================

def batch_process_scrna_doublets(
    input_folder: str,
    output_folder: str,
    method: str = "scrublet",
    expected_doublet_rate: float = 0.06,
    mad_threshold: float = 5.0
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    h5ad_files = [f for f in os.listdir(input_folder) if f.endswith(".h5ad")]
    if not h5ad_files:
        print("No H5AD files found.")
        return

    print(f"\n=== Starting batch doublet filtering ({len(h5ad_files)} files) ===")

    for f in sorted(h5ad_files):
        path_in = os.path.join(input_folder, f)
        print(f"\n>>> Processing {f}")
        adata = sc.read_h5ad(path_in)

        if method.lower() == "scrublet":
            adata_out = filter_doublets_scrublet(adata, expected_doublet_rate)
        else:
            adata_out = filter_doublets_simple(adata, mad_threshold)

        out_name = f.replace(".h5ad", "_singlets.h5ad")
        path_out = os.path.join(output_folder, out_name)
        adata_out.write(path_out, compression="gzip")
        print(f"  ✓ Saved filtered file: {path_out}")

    print("\n=== Batch processing complete ===")


# =============================================================================
# --- MAIN ---
# =============================================================================

if __name__ == "__main__":
    INPUT_DIR = "/mnt/f/H3K27/Data/scRNA"
    OUTPUT_DIR = "/mnt/f/H3K27/Data/scRNA"
    METHOD = "scrublet"
    EXPECTED_DOUBLET_RATE = 0.06
    MAD_THRESHOLD = 5.0

    batch_process_scrna_doublets(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        method=METHOD,
        expected_doublet_rate=EXPECTED_DOUBLET_RATE,
        mad_threshold=MAD_THRESHOLD,
    )
