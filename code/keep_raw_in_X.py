#!/usr/bin/env python3
"""
Set adata.X = adata.layers['raw_counts'] for an existing .h5ad file.

Use when you want the main matrix X to contain raw counts
(e.g. for pseudobulk / edgeR / peak→gene aggregation),
while still preserving other layers like 'tfidf'.
"""

import scanpy as sc
import anndata as ad
import numpy as np
from scipy.sparse import issparse, csr_matrix
from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Replace adata.X with adata.layers['raw_counts'] in a .h5ad file."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input .h5ad file (e.g. /mnt/f/H3K27/Data/scChi/1/scChi_1000k.h5ad)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=False,
        help=(
            "Path to output .h5ad file. "
            "If not provided, will write <input_basename>_rawX.h5ad in the same folder."
        ),
    )

    args = parser.parse_args()
    in_path = Path(args.input)

    if not in_path.exists():
        print(f"❌ ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Default output file: add _rawX before extension
    if args.output is None:
        out_path = in_path.with_name(in_path.stem + "_rawX.h5ad")
    else:
        out_path = Path(args.output)

    print("=" * 80)
    print(f"LOADING AnnData FROM: {in_path}")
    print("=" * 80)

    adata = sc.read_h5ad(in_path)

    print("\n[1] Original AnnData summary:")
    print(f"  Cells (n_obs): {adata.n_obs:,}")
    print(f"  Features (n_vars): {adata.n_vars:,}")
    print(f"  X dtype: {adata.X.dtype}")
    print(f"  X type:  {type(adata.X)}")
    print(f"  Layers:  {list(adata.layers.keys())}")
    print(f"  Has .raw: {adata.raw is not None}")

    if "raw_counts" not in adata.layers:
        print("❌ ERROR: 'raw_counts' layer not found in adata.layers.", file=sys.stderr)
        print("   Available layers:", list(adata.layers.keys()), file=sys.stderr)
        sys.exit(1)

    raw = adata.layers["raw_counts"]

    print("\n[2] raw_counts layer info:")
    print(f"  shape: {raw.shape}")
    print(f"  dtype: {raw.dtype}")
    print(f"  type:  {type(raw)}")

    # Ensure sparse CSR (best for big matrices)
    if not issparse(raw):
        print("  → raw_counts is dense; converting to CSR sparse matrix...")
        raw_csr = csr_matrix(raw)
    else:
        # Important: use .tocsr() to get consistent format
        raw_csr = raw.tocsr()

    # Quick sanity check on a tiny sample
    sample = raw_csr[:100, :100]
    sample_arr = sample.toarray()
    looks_integer = np.allclose(sample_arr, np.round(sample_arr))
    mean_val = sample_arr.mean()
    max_val = sample_arr.max()

    print("  ▶ raw_counts sample check (100×100):")
    print(f"    mean={mean_val:.4f}, max={max_val:.2f}, looks_integer={looks_integer}")

    # --- Set X = raw_counts ---
    print("\n[3] Replacing adata.X with raw_counts (sparse CSR)...")
    adata.X = raw_csr

    # You probably still want to keep tfidf and other layers for reference
    # so we do NOT touch adata.layers here.

    print("\n[4] New AnnData X summary:")
    print(f"  X dtype: {adata.X.dtype}")
    print(f"  X type:  {type(adata.X)}")
    if hasattr(adata.X, "nnz"):
        sparsity = 1.0 - (adata.X.nnz / adata.X.size)
        print(f"  X sparsity (fraction of zeros): {sparsity:.4f}")

    # Save to disk
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n[5] Saving updated AnnData...")
    print(f"  → {out_path}")
    adata.write_h5ad(out_path, compression="gzip")

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"\n✅ DONE. Output size: {size_mb:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
