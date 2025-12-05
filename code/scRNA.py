import pandas as pd
import numpy as np
import os
import sys
import anndata
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.io import mmread
import argparse
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
#                    MINIMAL scRNA-seq PREPROCESSING (QC ONLY)
# ============================================================================

def minimal_preprocess_rna(adata: anndata.AnnData,
                           min_genes: int = 100,
                           min_cells: int = 1,
                           target_sum: float = 1e4,
                           n_top_genes: int = None,
                           mt_prefix: str = 'mt-',
                           apply_normalization: bool = True) -> anndata.AnnData:
    """
    Minimal scRNA-seq preprocessing pipeline - QC and normalization ONLY.
    NO PCA, NO clustering, NO UMAP.
    
    This preserves maximum information for downstream integration.
    
    Args:
        adata: AnnData object with raw counts (Cells x Genes).
        min_genes: Minimum number of genes expressed in a cell.
        min_cells: Minimum number of cells expressing a gene.
        target_sum: Target sum for total-count normalization.
        n_top_genes: Number of highly variable genes to select (None = keep all).
        mt_prefix: Prefix for mitochondrial genes (used for QC).
        apply_normalization: Whether to apply normalization and log-transform.
    """
    print("\n" + "="*60)
    print("MINIMAL scRNA-seq PREPROCESSING (QC + NORMALIZATION ONLY)")
    print("="*60)

    # [1] Store raw counts
    print("\n[1/5] Storing raw counts...")
    adata.layers['raw_counts'] = adata.X.copy()
    print("✓ Raw counts saved in layers['raw_counts']")

    # [2] QC Metrics
    print(f"\n[2/5] Computing QC metrics...")
    # Identify mitochondrial genes and calculate their percentage
    adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    print("✓ QC metrics computed (total_counts, n_genes_by_counts, pct_counts_mt)")

    # [2.5] Optional cleanup before QC filtering
    print(f"\n[2.5] Removing unwanted features and empty cells...")

    # --- Remove obvious mouse genes (more specific regex) ---
    mask_human = ~adata.var_names.str.match(r'^(Mm|Gm|mouse)', case=False, na=False)
    removed_genes = adata.n_vars - mask_human.sum()
    adata = adata[:, mask_human].copy()
    print(f"✓ Removed {removed_genes:,} likely mouse features")

    # [3] Filtering (very conservative)
    print(f"\n[3/5] Quality control filtering (min_genes={min_genes}, min_cells={min_cells})...")
    print(f"Before: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print("\n[3.5/5] Cleaning feature names (removing 'hg19_' prefix)...")
    prefix_to_remove = 'hg19_'
    
    # Create clean gene names
    clean_var_names = [
        name.replace(prefix_to_remove, '') 
        for name in adata.var_names
    ]
    
    # Store original names for reference
    adata.var['original_var_names'] = adata.var_names.copy()
    
    # Update var_names to clean names BEFORE collapsing
    adata.var_names = clean_var_names
    adata.var_names_make_unique()  # Handle any duplicates in naming
    
    # --- Identify Duplicates and Prepare for Collapse ---
    unique_symbols = sorted(adata.var_names.unique())
    print(f"Total features remaining: {adata.n_vars:,}")
    print(f"Unique gene symbols found: {len(unique_symbols):,}")
    print(f"Features to be collapsed (duplicates): {adata.n_vars - len(unique_symbols):,}")
    
    # --- Collapse Features by Summing Counts (if duplicates exist) ---
    if adata.n_vars > len(unique_symbols):
        print(f"\n[3.5] Collapsing duplicate features...")
        adata.X = adata.X.tocsr()  # ensure CSR format
        df = pd.DataFrame.sparse.from_spmatrix(adata.X, columns=adata.var_names)
        df = df.groupby(df.columns, axis=1).sum()
        
        # Preserve var metadata by taking first occurrence of each gene
        var_collapsed = adata.var.groupby(adata.var_names).first()
        
        adata = anndata.AnnData(
            X=csr_matrix(df.sparse.to_coo()), 
            obs=adata.obs.copy(),
            var=var_collapsed
        )
        adata.var_names = df.columns
        print(f"✓ Feature collapse complete. New feature count: {adata.n_vars:,}")
    else:
        print(f"\n[3.5] No duplicate features to collapse.")

    print(f"After filtering: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"✓ QC filtering complete (very lenient thresholds for rare populations)")

    # [4] Normalization & Log-transformation (Optional)
    if apply_normalization:
        print(f"\n[4/5] Normalization (target_sum={target_sum}) and log-transformation...")
        # Normalize count data to a target sum (e.g., 10,000 counts per cell)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        # Log-transform the data
        sc.pp.log1p(adata)
        #adata.layers['log_normalized'] = adata.X.copy()
        print("✓ Normalization and log-transformation complete")
        #print("✓ Log-normalized data saved in layers['log_normalized'] and .X")
    else:
        print(f"\n[4/5] Skipping normalization")

    # [5] Highly Variable Gene Selection (Optional)
    if n_top_genes is not None:
        print(f"\n[5/5] Highly Variable Gene selection (n_top_genes={n_top_genes})...")
        
        # Identify highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')
        
        n_hvg = adata.var['highly_variable'].sum()
        print(f"✓ Identified {n_hvg:,} highly variable genes")
        print(f"  Note: Full data preserved in .raw for later use")
        
        # Subset to highly variable genes
        adata = adata[:, adata.var.highly_variable].copy()
        print(f"✓ Subsetted to {adata.n_vars:,} highly variable genes")

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE - NO DIMENSIONALITY REDUCTION")
    print("="*60)
    print(f"Final shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print("\nLayers available:")
    for layer in adata.layers.keys():
        print(f"  - {layer}")
    if adata.raw is not None:
        print(f"\n.raw contains: {adata.raw.n_vars:,} genes (full dataset)")

    return adata


# ============================================================================
#                             LOAD AND PROCESS MTX
# ============================================================================

def process_mtx(matrix_file: str,
                barcodes_file: str,
                features_file: str,
                output_path: str = None,
                **kwargs):
    """
    Load and process 10x Genomics MTX format files for scRNA-seq.

    The 10x MTX format is assumed to be (Features x Cells).
    """
    print(f"\n{'='*60}")
    print("LOADING MTX FORMAT scRNA-seq DATA")
    print(f"{'='*60}")

    # Load MTX (Reads into COO format by default)
    try:
        matrix_coo = mmread(matrix_file)
    except FileNotFoundError:
        print(f"Error: Matrix file not found at {matrix_file}")
        sys.exit(1)

    # 1. Convert to CSR and transpose: (Features x Cells) -> (Cells x Features)
    print("Converting matrix to CSR format and transposing...")
    matrix = matrix_coo.T.tocsr()

    # Load cell (obs) and gene (var) names
    try:
        print("Loading barcodes and features...")
        cell_names = pd.read_csv(barcodes_file, header=None, compression='gzip', sep='\t').iloc[:, 0].values
        
        # The 10x features file has two or three columns
        feature_df = pd.read_csv(features_file, header=None, compression='gzip', sep='\t')
        
        # Try to get gene symbols (usually column 1, index 1)
        if feature_df.shape[1] >= 2:
            gene_names = feature_df.iloc[:, 1].values  # Usually the gene symbol
            feature_metadata = feature_df.iloc[:, [0, 1]].copy()
            feature_metadata.columns = ['Ensembl_ID', 'Gene_Symbol']
            if feature_df.shape[1] >= 3:
                feature_metadata['Feature_Type'] = feature_df.iloc[:, 2].values
            feature_metadata.set_index('Gene_Symbol', inplace=True)
        else:
            # Fallback if only one column
            gene_names = feature_df.iloc[:, 0].values
            feature_metadata = pd.DataFrame(index=gene_names)
            
    except Exception as e:
        print(f"Error loading barcodes or features: {e}")
        sys.exit(1)

    # Create AnnData
    print("Creating AnnData object...")
    adata = anndata.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=cell_names),
        var=feature_metadata
    )

    print(f"✓ Initial AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"  Cell Barcode Example: {adata.obs_names[0]}")
    print(f"  Gene Symbol Example: {adata.var_names[0]}")

    # Minimal preprocessing (QC only, no dimensionality reduction)
    adata = minimal_preprocess_rna(
        adata,
        min_genes=kwargs.get('min_genes', 100),
        min_cells=kwargs.get('min_cells', 1),
        target_sum=kwargs.get('target_sum', 1e4),
        n_top_genes=kwargs.get('n_top_genes', None),
        mt_prefix=kwargs.get('mt_prefix', 'mt-'),
        apply_normalization=kwargs.get('apply_normalization', True)
    )

    # Save
    if output_path:
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        adata.write_h5ad(output_path, compression='gzip')
        print(f"✓ Saved to: {output_path}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Final: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print("\nThis h5ad contains:")
    print("  - Raw counts (layers['raw_counts'])")
    if kwargs.get('apply_normalization', True):
        print("  - Log-normalized (layers['log_normalized'] and .X)")
    if kwargs.get('n_top_genes') is not None:
        print("  - Highly variable genes (subset)")
        print("  - Full dataset (.raw)")
    print("  - QC metrics (obs and var)")
    print("\nReady for integration and downstream analysis!")

    return adata


# ============================================================================
#                                CLI INTERFACE
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="scRNA-seq Data Processor - Raw Data Preservation Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 10x MTX format (keep all genes)
  python scRNA.py --matrix matrix.mtx.gz --barcodes barcodes.tsv.gz --features features.tsv.gz --output rna_out.h5ad
  
  # With highly variable gene selection
  python scRNA.py --matrix matrix.mtx.gz --barcodes barcodes.tsv.gz --features features.tsv.gz --n-top-genes 20000 --output rna_out.h5ad
  
  # Very lenient filtering for rare populations
  python scRNA.py --matrix matrix.mtx.gz --barcodes barcodes.tsv.gz --features features.tsv.gz --min-genes 50 --min-cells 1 --output rna_out.h5ad
        """
    )

    # Required arguments
    parser.add_argument('--matrix', required=True, help="Input matrix.mtx.gz")
    parser.add_argument('--barcodes', required=True, help="Input barcodes.tsv.gz")
    parser.add_argument('--features', required=True, help="Input features.tsv.gz")
    parser.add_argument('--output', required=True, help="Output .h5ad file")

    # QC parameters (very conservative defaults for rare populations)
    parser.add_argument('--min-genes', type=int, default=100,
                        help="Minimum genes per cell (default: 100, lenient)")
    parser.add_argument('--min-cells', type=int, default=1,
                        help="Minimum cells per gene (default: 1, keep all genes)")
    
    # Normalization parameters
    parser.add_argument('--target-sum', type=float, default=10000.0,
                        help="Normalization target sum (default: 10000.0)")
    parser.add_argument('--no-normalization', action='store_true',
                        help="Skip normalization and log-transformation")
    
    # Feature selection (optional - None means keep all)
    parser.add_argument('--n-top-genes', type=int, default=None,
                        help="Number of highly variable genes to select (default: None, keep all)")
    
    # QC parameters
    parser.add_argument('--mt-prefix', type=str, default='mt-',
                        help="Mitochondrial gene prefix for QC (default: 'mt-', use 'MT-' for human)")

    args = parser.parse_args()

    # Validate inputs
    required = [args.matrix, args.barcodes, args.features]
    if not all(required):
        parser.error("--matrix, --barcodes, and --features required")
    for path in required:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)

    # Process
    kwargs = {
        'min_genes': args.min_genes,
        'min_cells': args.min_cells,
        'target_sum': args.target_sum,
        'n_top_genes': args.n_top_genes,
        'mt_prefix': args.mt_prefix,
        'apply_normalization': not args.no_normalization,
    }

    process_mtx(
        matrix_file=args.matrix,
        barcodes_file=args.barcodes,
        features_file=args.features,
        output_path=args.output,
        **kwargs
    )


if __name__ == "__main__":
    main()