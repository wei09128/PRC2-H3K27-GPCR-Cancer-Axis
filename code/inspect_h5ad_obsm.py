import anndata
import sys
from pathlib import Path
import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse # Necessary for checking and converting sparse matrix types
from scipy.sparse import csr_matrix, issparse

def calculate_density(matrix, name):
    """
    Calculates and prints the percentage of non-zero elements (data density) 
    for the given matrix.
    """
    if matrix is None or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        print(f"\nMatrix density calculation skipped: '{name}' is empty or None.")
        return

    total_elements = matrix.shape[0] * matrix.shape[1]
    
    if total_elements == 0:
        print(f"\nMatrix density calculation skipped: '{name}' has zero total elements.")
        return

    try:
        # Use .nnz attribute which is available on sparse matrices (like those in AnnData)
        if hasattr(matrix, 'nnz'):
            non_zero_elements = matrix.nnz
        else:
            # Fallback: Convert to CSR sparse matrix to efficiently get non_zero_elements
            sparse_matrix = scipy.sparse.csr_matrix(matrix)
            non_zero_elements = sparse_matrix.nnz
            
        percentage_non_zero = (non_zero_elements / total_elements) * 100
        
        print(f"\n--- Matrix Density for {name} ---")
        print(f"Non-zero elements: {non_zero_elements:,}")
        print(f"Total elements: {total_elements:,}")
        print(f"Percentage of Non-Zero Values (Density): {percentage_non_zero:.4f}%")
        
    except Exception as e:
        print(f"\nError calculating density for '{name}': {e}")


def inspect_anndata_file(filepath):
    """
    Loads an H5AD file and prints metadata, previews the expression matrices,
    and calculates the percentage of non-zero values (data density).
    """
    file_path = Path(filepath)

    if not file_path.exists():
        print(f"ERROR: File not found at path: {filepath}")
        return

    try:
        # Load the AnnData object
        print(f"Loading AnnData object from: {file_path.name}")
        adata = anndata.read_h5ad(file_path, backed='r') # Read-only, backed mode
        
        print(adata.obsm_keys())
    except Exception as e:
        print(f"An unexpected error occurred while processing the file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_h5ad.py <path_to_h5ad_file>")
        sys.exit(1)
        
    h5ad_filepath = sys.argv[1]
    # Ensure numpy and scipy.sparse are imported before running the function
    # Note: These imports are already at the top, but keeping this block as a safety measure
    if 'numpy' not in sys.modules:
        import numpy as np
    if 'scipy.sparse' not in sys.modules:
        import scipy.sparse
        
    inspect_anndata_file(h5ad_filepath)