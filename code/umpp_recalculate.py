import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
import os
import argparse # Import argparse for command-line execution

def run_processing_pipeline(input_h5ad: str, output_h5ad: str, batch_key: str):
    """
    Performs standard single-cell processing: filtering, scaling, PCA, Harmony, UMAP, and Leiden.
    
    Args:
        input_h5ad: Path to the input AnnData file (e.g., gene accessibility matrix).
        output_h5ad: Path to save the final processed AnnData file.
        batch_key: The column name in adata.obs to use for Harmony batch correction.
    """
    
    OUTPUT_DIR = os.path.dirname(output_h5ad)
    if not OUTPUT_DIR:
        OUTPUT_DIR = "."
    
    print("=" * 70)
    print("SC GENE ACCESSIBILITY PROCESSING PIPELINE")
    print("=" * 70)
    print(f"Input H5AD: {input_h5ad}")
    print(f"Output H5AD: {output_h5ad}")
    print(f"Batch Key: {batch_key}")
    
    # --- 1. Load the data ---
    try:
        adata = sc.read_h5ad(input_h5ad)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_h5ad}. Exiting.")
        return
        
    print(f"\nLoaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Check if batch key exists
    if batch_key not in adata.obs.columns:
        print(f"ERROR: Batch key '{batch_key}' not found in adata.obs")
        print(f"Available columns: {list(adata.obs.columns)}")
        return

    print("\n" + "=" * 70)
    print("PREPROCESSING")
    print("=" * 70)

    # Store raw counts
    print("Storing raw accessibility counts...")
    adata.layers['raw_accessibility'] = adata.X.copy()

    # Filter genes (very lenient for accessibility data)
    print(f"\nFiltering genes present in >= 3 cells...")
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  Remaining: {adata.n_vars:,} genes")

    # Select highly variable features
    print("\nSelecting highly variable genes...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=3000,
        flavor='seurat_v3',
        subset=False
    )
    n_hvg = adata.var.highly_variable.sum()
    print(f"  ✓ {n_hvg:,} HVGs identified")

    # Subset to HVGs
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"  ✓ Subsetted to {adata.n_vars:,} genes")

    # Scale (optional but recommended for accessibility data)
    print("\nScaling...")
    sc.pp.scale(adata, max_value=10)

    # PCA
    print("\nRunning PCA (n_comps=50)...")
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    print("  ✓ PCA complete")

    print("\n" + "=" * 70)
    print("HARMONY BATCH CORRECTION")
    print("=" * 70)

    try:
        # Extract PCA coordinates and metadata
        pca_df = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names)
        meta_df = adata.obs[[batch_key]].copy()
        
        # Run Harmony
        print(f"Running Harmony on '{batch_key}'...")
        harmony_out = hm.run_harmony(
            pca_df,
            meta_df,
            vars_use=[batch_key],
            max_iter_harmony=100,
            verbose=True
        )
        
        # Store corrected embeddings
        adata.obsm['X_harmony'] = harmony_out.Z_corr.T
        
        print(f"  ✓ Harmony complete")
        print(f"  Output shape: {adata.obsm['X_harmony'].shape}")
        use_rep = 'X_harmony'
        
    except Exception as e:
        print(f"  ✗ Harmony failed: {e}")
        print("  → Using uncorrected PCA")
        use_rep = 'X_pca'

    print("\n" + "=" * 70)
    print("CLUSTERING & UMAP")
    print("=" * 70)

    # Compute neighbors
    print(f"Computing neighbors using {use_rep}...")
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=30, n_pcs=50)

    # UMAP
    print("Computing UMAP...")
    sc.tl.umap(adata)

    # Leiden clustering with multiple resolutions
    print("\nComputing Leiden clustering...")
    for res in [0.1, 0.5, 1.0]:
        key = f'leiden_res_{res}'
        sc.tl.leiden(adata, resolution=res, key_added=key)
        n_clusters = adata.obs[key].nunique()
        print(f"  Resolution {res}: {n_clusters} clusters")

    # Also add default leiden (res 0.5)
    sc.tl.leiden(adata, resolution=0.5, key_added='leiden')

    print("\n" + "=" * 70)
    print("SAVING PLOTS")
    print("=" * 70)

    # Ensure output directory exists and set scanpy figdir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sc.settings.figdir = OUTPUT_DIR
    
    # Plot different resolutions
    for res in [0.1, 0.5, 1.0]:
        key = f'leiden_res_{res}'
        sc.pl.umap(
            adata,
            color=key,
            title=f'UMAP - Leiden Resolution {res}',
            save=f'_{key}.png',
            show=False,
            frameon=False
        )
        print(f"  ✓ Saved UMAP for resolution {res}")

    # Plot by batch
    sc.pl.umap(
        adata,
        color=batch_key,
        title=f'UMAP colored by {batch_key}',
        save=f'_{batch_key}.png',
        show=False,
        frameon=False
    )
    print(f"  ✓ Saved UMAP colored by {batch_key}")

    print("\n" + "=" * 70)
    print("SAVING FINAL ANNDATA")
    print("=" * 70)

    # Save
    adata.write(output_h5ad)
    print(f"✅ Success! Saved to: {output_h5ad}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"Batch correction: {use_rep}")
    print(f"UMAP coordinates: adata.obsm['X_umap']")
    print(f"Clustering keys:")
    for res in [0.1, 0.5, 1.0]:
        key = f'leiden_res_{res}'
        n_clusters = adata.obs[key].nunique()
        print(f"  - {key}: {n_clusters} clusters")


def main():
    """Parses command line arguments and initiates the processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Run scRNA/scATAC/scChIP post-processing (PCA, Harmony, UMAP, Leiden) on an AnnData file."
    )
    
    # Required arguments
    parser.add_argument(
        '--input_h5ad',
        type=str,
        required=True,
        help="Path to the input AnnData file (e.g., scChi_accessibility.h5ad)."
    )
    parser.add_argument(
        '--output_h5ad',
        type=str,
        required=True,
        help="Path to save the final processed AnnData file (e.g., scChi_accessibility_processed.h5ad)."
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--batch_key',
        type=str,
        default='Batch_ID',
        help="The column name in adata.obs to use for Harmony batch correction (Default: Batch_ID)."
    )
    
    args = parser.parse_args()
    
    # Call the processing function with parsed arguments
    run_processing_pipeline(
        input_h5ad=args.input_h5ad,
        output_h5ad=args.output_h5ad,
        batch_key=args.batch_key
    )


if __name__ == "__main__":
    main()
