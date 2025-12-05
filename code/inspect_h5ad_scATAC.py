#!/usr/bin/env python3
"""
AnnData Inspection Script
Loads an AnnData object and prints key metrics about the cell-wise (row)
data distribution, sparsity, layers, and batch distribution before any transformation.

Usage examples:
  python inspect_h5ad_scATAC.py /path/to/file.h5ad
  python inspect_h5ad_scATAC.py /path/to/file.h5ad --batch_key Condition
  python inspect_h5ad_scATAC.py /path/to/file.h5ad --plot_dir ./my_plots
"""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# --- DEFAULT CONFIGURATION (used if args not provided) ---
DEFAULT_INPUT_PATH = '/mnt/f/H3K27/Data/scATAC/1/scATAC_1000k_fixed2_rawX.h5ad'
DEFAULT_INSPECTION_PLOT_DIR = './inspection_plots'
DEFAULT_BATCH_KEY = 'Batch_ID'
# ----------------------------------------------------------


def inspect_adata(input_path: str, plot_dir: str, batch_key: str):
    """Loads and inspects the AnnData object."""

    print("=" * 80)
    print(f"LOADING AND INSPECTING DATA: {input_path}")
    print("=" * 80)

    try:
        adata = sc.read_h5ad(input_path)
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = plot_dir
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        return

    # --- 1. Basic Dimensions and Type ---
    print("\n[1] AnnData Summary:")
    print(f"  Cells (n_obs): {adata.n_obs:,}")
    print(f"  Features/Genes (n_vars): {adata.n_vars:,}")
    print(f"  X Data Type: {adata.X.dtype}")
    print(f"  X Format: {type(adata.X)}")
    print(f"  Has .raw: {adata.raw is not None}")

    # --- 1b. Layers Summary ---
    print("\n[1b] Layers summary:")
    if len(adata.layers.keys()) == 0:
        print("  (no layers present)")
    else:
        print(f"  Available layers: {list(adata.layers.keys())}")
        for lname in adata.layers.keys():
            layer = adata.layers[lname]
            # Handle sparse vs dense
            if hasattr(layer, "nnz"):  # sparse matrix
                nnz = layer.nnz
                total = layer.shape[0] * layer.shape[1]
                sparsity_layer = 1.0 - (nnz / total)
                dtype = layer.dtype
            else:
                total = layer.size
                zeros = (layer == 0).sum()
                sparsity_layer = zeros / total
                dtype = layer.dtype
            print(
                f"  - Layer '{lname}': shape={layer.shape}, "
                f"dtype={dtype}, sparsity={sparsity_layer:.4f}"
            )

        # If raw_counts exists, peek at its basic stats
        if "raw_counts" in adata.layers:
            rc = adata.layers["raw_counts"]
            print("\n  ▶ raw_counts quick check:")
            if hasattr(rc, "nnz"):
                nnz = rc.nnz
                total = rc.shape[0] * rc.shape[1]
                sparsity_rc = 1.0 - (nnz / total)
                sample = rc[:100, :100].toarray()
            else:
                total = rc.size
                zeros = (rc == 0).sum()
                sparsity_rc = zeros / total
                sample = rc[:100, :100]
            print(f"    shape={rc.shape}, dtype={rc.dtype}, sparsity={sparsity_rc:.4f}")
            print(f"    sample mean={sample.mean():.4f}, max={sample.max():.2f}")
            print(f"    looks_integer={np.allclose(sample, np.round(sample))}")

    # If .raw exists, summarize it too
    if adata.raw is not None:
        print("\n[1c] .raw summary:")
        print(f"  .raw shape: {adata.raw.n_obs:,} × {adata.raw.n_vars:,}")
        print(f"  .raw dtype: {adata.raw.X.dtype}")
        if hasattr(adata.raw.X, "nnz"):
            nnz_raw = adata.raw.X.nnz
            total_raw = adata.raw.X.shape[0] * adata.raw.X.shape[1]
            sparsity_raw = 1.0 - (nnz_raw / total_raw)
        else:
            total_raw = adata.raw.X.size
            zeros_raw = (adata.raw.X == 0).sum()
            sparsity_raw = zeros_raw / total_raw
        print(f"  .raw sparsity (Fraction of Zeros): {sparsity_raw:.4f}")

    # --- 2. Sparsity of X ---
    if hasattr(adata.X, 'nnz'):
        if adata.X.nnz == 0:
            sparsity = 1.0
        else:
            sparsity = 1.0 - (adata.X.nnz / adata.X.size)
    else:
        sparsity = (adata.X == 0).sum() / adata.X.size

    print("\n[2] X Sparsity:")
    print(f"  Sparsity (Fraction of Zeros): {sparsity:.4f}")

    # --- 3. Row/Cell Metrics (Total Counts) ---
    print("\n[3] Cell-Wise Metrics (Library Size/Total Counts):")
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=(50,))

    total_counts = adata.obs['total_counts']
    print(f"  Total Counts (Min): {total_counts.min():,.2f}")
    print(f"  Total Counts (Median): {total_counts.median():,.2f}")
    print(f"  Total Counts (Mean): {total_counts.mean():,.2f}")
    print(f"  Total Counts (Max): {total_counts.max():,.2f}")

    # Plot distribution of library size
    plot_filename = 'total_counts_pre_transform.png'
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(total_counts, bins=50, log=True, color='gray', edgecolor='black')
    ax.set_title("Distribution of Total Accessibility per Cell")
    ax.set_xlabel("Total Accessibility Score (log scale)")
    ax.set_ylabel("Number of Cells (log scale)")
    fig.savefig(Path(plot_dir) / plot_filename)
    plt.close(fig)
    print(f"  -> Saved total counts histogram to: {plot_dir}/{plot_filename}")

    # --- 4. Batch Distribution ---
    print(f"\n[4] Batch Distribution (Key: '{batch_key}'):")
    if batch_key in adata.obs.columns:
        if not pd.api.types.is_categorical_dtype(adata.obs[batch_key]):
            adata.obs[batch_key] = adata.obs[batch_key].astype('category')

        batch_dist = adata.obs[batch_key].value_counts(normalize=False)
        batch_pct = adata.obs[batch_key].value_counts(normalize=True) * 100

        for batch_id, count in batch_dist.items():
            print(f"  - Batch {batch_id}: {count:,} cells ({batch_pct[batch_id]:.2f}%)")
    else:
        print(f"  ⚠ Batch key '{batch_key}' not found in adata.obs.")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inspect an AnnData .h5ad file (sparsity, layers, QC, batch distribution)."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=f"Path to .h5ad file (default: {DEFAULT_INPUT_PATH})"
    )
    parser.add_argument(
        "--plot_dir",
        "-o",
        default=DEFAULT_INSPECTION_PLOT_DIR,
        help=f"Directory to save plots (default: {DEFAULT_INSPECTION_PLOT_DIR})"
    )
    parser.add_argument(
        "--batch_key",
        "-b",
        default=DEFAULT_BATCH_KEY,
        help=f"obs column to use as batch key (default: {DEFAULT_BATCH_KEY})"
    )

    args = parser.parse_args()

    inspect_adata(args.input_path, args.plot_dir, args.batch_key)
