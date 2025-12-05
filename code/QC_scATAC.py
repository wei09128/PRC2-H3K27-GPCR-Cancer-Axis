#!/usr/bin/env python3
"""
Redo scATAC QC on peak-level AnnData and aggregate peaks to genes
using a gene annotation GTF (TSS ± 2kb window).

Requirements:
    pip install scanpy anndata pandas numpy scipy pyranges

Example usage:

    python scATAC_qc_and_peak2gene_from_gtf.py \
      --input_h5ad /mnt/f/H3K27/Data/scATAC/scATAC_peaks.h5ad \
      --gtf /mnt/f/H3K27/Data/reference/gencode.v43.annotation.gtf \
      --output_qc_h5ad /mnt/f/H3K27/Data/scATAC/scATAC_peaks_QC.h5ad \
      --output_gene_h5ad /mnt/f/H3K27/Data/scATAC/scATAC_gene_QC.h5ad \
      --min_counts 1000 \
      --min_peaks 500 \
      --min_cells_per_peak 5 \
      --tss_window 2000
"""

import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import pyranges as pr


# ---------------------------------------------------------------------
# QC helpers
# ---------------------------------------------------------------------

def compute_basic_qc(adata):
    """
    Compute basic QC metrics for scATAC:
      - total_counts (per cell)
      - n_peaks (non-zero peaks per cell)
      - n_cells_by_peak (non-zero cells per peak)
    """
    print("Computing basic QC metrics...")
    X = adata.X

    if sp.issparse(X):
        adata.obs["total_counts"] = np.asarray(X.sum(axis=1)).ravel()
        adata.obs["n_peaks"] = np.asarray((X > 0).sum(axis=1)).ravel()
        adata.var["n_cells_by_peak"] = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        adata.obs["total_counts"] = X.sum(axis=1)
        adata.obs["n_peaks"] = (X > 0).sum(axis=1)
        adata.var["n_cells_by_peak"] = (X > 0).sum(axis=0)

    print("  Example cell QC head:")
    print(adata.obs[["total_counts", "n_peaks"]].head())
    print("  Example peak QC head:")
    print(adata.var[["n_cells_by_peak"]].head())


def apply_qc_filters(
    adata,
    min_counts=None,
    max_counts=None,
    min_peaks=None,
    min_cells_per_peak=None,
):
    """
    Apply QC filters to cells and peaks.
    Returns a new filtered AnnData.
    """
    print("\nApplying QC filters...")

    # ---- Cell filters ----
    cell_mask = np.ones(adata.n_obs, dtype=bool)

    if min_counts is not None:
        cell_mask &= adata.obs["total_counts"] >= min_counts
        print(f"  Filter: cells with total_counts >= {min_counts}")

    if max_counts is not None:
        cell_mask &= adata.obs["total_counts"] <= max_counts
        print(f"  Filter: cells with total_counts <= {max_counts}")

    if min_peaks is not None:
        cell_mask &= adata.obs["n_peaks"] >= min_peaks
        print(f"  Filter: cells with n_peaks >= {min_peaks}")

    print(f"  Cells before filter: {adata.n_obs}")
    adata = adata[cell_mask].copy()
    print(f"  Cells after filter:  {adata.n_obs}")

    # ---- Peak filters ----
    peak_mask = np.ones(adata.n_vars, dtype=bool)

    if min_cells_per_peak is not None:
        peak_mask &= adata.var["n_cells_by_peak"] >= min_cells_per_peak
        print(f"  Filter: peaks present in >= {min_cells_per_peak} cells")

    print(f"  Peaks before filter: {adata.n_vars}")
    adata = adata[:, peak_mask].copy()
    print(f"  Peaks after filter:  {adata.n_vars}")

    return adata


# ---------------------------------------------------------------------
# Build peak → gene map from GTF
# ---------------------------------------------------------------------

def build_peak_to_gene_from_gtf(
    atac_adata,
    gtf_path,
    tss_window=2000,
    peak_coord_col=None,
):
    """
    Build a peak→gene mapping from:
      - atac_adata.var_names or a column with genomic coords
      - gene annotation GTF (TSS ± tss_window)

    Assumes peak coordinates look like: "chr1:12345-12500".
    If stored in a column (e.g., adata.var['peak']), set peak_coord_col.
    """

    print("\nBuilding peak→gene mapping from GTF...")
    # --- 1. Get peaks as a DataFrame with chrom, start, end ---
    if peak_coord_col is None:
        peak_strings = atac_adata.var_names.astype(str)
    else:
        peak_strings = atac_adata.var[peak_coord_col].astype(str)

    # split "chr1:1000-2000"
    peak_df = peak_strings.str.replace("chr", "", regex=False).str.split("[:-]", expand=True)
    if peak_df.shape[1] != 3:
        raise ValueError(
            "Could not parse peak coordinates. "
            "Expected format like 'chr1:1000-2000'."
        )

    peak_df.columns = ["Chromosome", "Start", "End"]
    peak_df["Start"] = peak_df["Start"].astype(int)
    peak_df["End"] = peak_df["End"].astype(int)
    peak_df["Chromosome"] = "chr" + peak_df["Chromosome"].astype(str)  # add chr back
    peak_df["Peak_ID"] = atac_adata.var_names.astype(str)

    peaks_gr = pr.PyRanges(peak_df[["Chromosome", "Start", "End", "Peak_ID"]])

    # --- 2. Read GTF and create TSS windows ---
    print(f"  Reading GTF from: {gtf_path}")
    gtf = pr.read_gtf(gtf_path)

    # Keep only 'gene' entries and needed columns
    genes = gtf[gtf.Feature == "gene"]
    genes_df = genes.df

    # Gene name column can be 'gene_name' (common in Gencode)
    if "gene_name" in genes_df.columns:
        gene_name_col = "gene_name"
    elif "Name" in genes_df.columns:
        gene_name_col = "Name"
    elif "gene_id" in genes_df.columns:
        gene_name_col = "gene_id"
    else:
        raise ValueError("Could not find gene name column in GTF (tried gene_name, Name, gene_id).")

    genes_df = genes_df[["Chromosome", "Start", "End", "Strand", gene_name_col]].copy()
    genes_df.rename(columns={gene_name_col: "gene"}, inplace=True)

    # Compute TSS
    print(f"  Creating TSS windows ±{tss_window} bp...")
    tss = genes_df.copy()
    # For + strand: TSS = Start; for - strand: TSS = End
    plus_mask = tss["Strand"] == "+"
    minus_mask = tss["Strand"] == "-"

    tss_start = tss["Start"].copy()
    tss_end = tss["End"].copy()

    tss_start[plus_mask] = tss["Start"][plus_mask] - tss_window
    tss_end[plus_mask] = tss["Start"][plus_mask] + tss_window

    tss_start[minus_mask] = tss["End"][minus_mask] - tss_window
    tss_end[minus_mask] = tss["End"][minus_mask] + tss_window

    # Ensure non-negative starts
    tss_start = tss_start.clip(lower=0)

    tss_promoters = pd.DataFrame({
        "Chromosome": tss["Chromosome"],
        "Start": tss_start.astype(int),
        "End": tss_end.astype(int),
        "gene": tss["gene"].astype(str),
    })

    promoters_gr = pr.PyRanges(tss_promoters)

    # --- 3. Overlap peaks with TSS windows ---
    print("  Intersecting peaks with promoter/TSS windows...")
    overlaps = peaks_gr.join(promoters_gr)

    if overlaps.df.empty:
        raise ValueError("No overlaps between peaks and promoters; check coordinate formats and tss_window.")

    ov = overlaps.df[["Peak_ID", "gene"]].drop_duplicates().reset_index(drop=True)
    print(f"  Got {len(ov)} peak→gene links after overlap.")

    return ov  # DataFrame with Peak_ID, gene


# ---------------------------------------------------------------------
# Aggregate peaks → genes
# ---------------------------------------------------------------------

def aggregate_peaks_to_genes(atac_adata, peak_to_gene_df):
    """
    Aggregate peak-level counts to gene-level counts using a mapping DF:
      columns = ['Peak_ID', 'gene'].
    """

    print("\nAggregating peaks to genes...")

    peak_index = pd.Index(atac_adata.var_names.astype(str))
    gene_names = np.array(sorted(peak_to_gene_df["gene"].unique()))
    gene_index = pd.Index(gene_names)

    # map peak IDs and gene names to indices
    peak_idx = peak_index.get_indexer(peak_to_gene_df["Peak_ID"].astype(str))
    valid_mask = peak_idx >= 0
    peak_to_gene_df = peak_to_gene_df.loc[valid_mask].reset_index(drop=True)
    peak_idx = peak_idx[valid_mask]

    gene_idx = gene_index.get_indexer(peak_to_gene_df["gene"].astype(str))

    n_peaks = atac_adata.n_vars
    n_genes = gene_index.size

    # weight = 1 for all links (can change later)
    weights = np.ones(len(peak_idx), dtype=float)

    import scipy.sparse as sp
    P2G = sp.csr_matrix(
        (weights, (peak_idx, gene_idx)),
        shape=(n_peaks, n_genes),
    )

    X = atac_adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    print("  Multiplying (cells x peaks) @ (peaks x genes) ...")
    X_gene = X @ P2G

    gene_adata = ad.AnnData(
        X_gene,
        obs=atac_adata.obs.copy(),
        var=pd.DataFrame(index=gene_index),
    )
    gene_adata.var_names.name = "gene"

    print("  Gene-level AnnData shape:", gene_adata.shape)
    return gene_adata


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Redo scATAC QC and peak-to-gene aggregation using GTF (TSS ± window)."
    )
    parser.add_argument("--input_h5ad", type=str, required=True,
                        help="Input peak-level scATAC AnnData (.h5ad), expected cells x peaks.")
    parser.add_argument("--gtf", type=str, required=True,
                        help="Gene annotation GTF file (e.g., Gencode).")
    parser.add_argument("--output_qc_h5ad", type=str, required=True,
                        help="Output path for QC'd peak-level AnnData.")
    parser.add_argument("--output_gene_h5ad", type=str, required=True,
                        help="Output path for gene-level aggregated AnnData.")
    parser.add_argument("--min_counts", type=float, default=1000,
                        help="Minimum total counts per cell (default: 1000).")
    parser.add_argument("--max_counts", type=float, default=None,
                        help="Maximum total counts per cell (default: None).")
    parser.add_argument("--min_peaks", type=int, default=500,
                        help="Minimum detected peaks per cell (default: 500).")
    parser.add_argument("--min_cells_per_peak", type=int, default=5,
                        help="Minimum number of cells per peak (default: 5).")
    parser.add_argument("--tss_window", type=int, default=2000,
                        help="TSS window (bp) upstream/downstream (default: 2000).")
    parser.add_argument("--peak_coord_col", type=str, default=None,
                        help="If peak coords are in adata.var[column], provide its name; "
                             "otherwise var_names are used.")

    args = parser.parse_args()

    # ---- Load scATAC peaks ----
    print(f"Loading peak-level AnnData from: {args.input_h5ad}")
    atac_adata = sc.read_h5ad(args.input_h5ad)
    print("Original shape (cells x peaks?):", atac_adata.shape)

    # If your object is peaks x cells, uncomment this:
    # atac_adata = atac_adata.T
    # print("Transposed AnnData; new shape:", atac_adata.shape)

    # ---- QC ----
    compute_basic_qc(atac_adata)

    atac_adata_qc = apply_qc_filters(
        atac_adata,
        min_counts=args.min_counts,
        max_counts=args.max_counts,
        min_peaks=args.min_peaks,
        min_cells_per_peak=args.min_cells_per_peak,
    )

    # recompute QC metrics after filtering
    compute_basic_qc(atac_adata_qc)

    os.makedirs(os.path.dirname(args.output_qc_h5ad), exist_ok=True)
    print(f"\nSaving QC'd peak-level AnnData to: {args.output_qc_h5ad}")
    atac_adata_qc.write(args.output_qc_h5ad)

    # ---- Build peak→gene map from GTF ----
    peak_to_gene_df = build_peak_to_gene_from_gtf(
        atac_adata_qc,
        gtf_path=args.gtf,
        tss_window=args.tss_window,
        peak_coord_col=args.peak_coord_col,
    )

    # ---- Aggregate peaks → genes ----
    gene_adata = aggregate_peaks_to_genes(atac_adata_qc, peak_to_gene_df)

    os.makedirs(os.path.dirname(args.output_gene_h5ad), exist_ok=True)
    print(f"Saving gene-level AnnData to: {args.output_gene_h5ad}")
    gene_adata.write(args.output_gene_h5ad)

    print("\nDone.")


if __name__ == "__main__":
    main()
