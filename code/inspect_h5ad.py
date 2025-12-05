import sys
from pathlib import Path

import anndata
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import issparse


def calculate_density(matrix, name):
    """
    Safely calculates and prints the percentage of non-zero elements (data density)
    for the given matrix. Skips if matrix is non-numeric or oddly shaped.
    """
    if matrix is None:
        print(f"\nMatrix density calculation skipped: '{name}' is None.")
        return

    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        print(f"\nMatrix density calculation skipped: '{name}' has invalid shape {shape}.")
        return

    n_rows, n_cols = shape
    if n_rows == 0 or n_cols == 0:
        print(f"\nMatrix density calculation skipped: '{name}' has zero rows or columns.")
        return

    # Check dtype if available
    dtype = getattr(matrix, "dtype", None)
    if dtype is not None and not np.issubdtype(dtype, np.number):
        print(
            f"\nMatrix density calculation skipped: '{name}' has non-numeric dtype {dtype}. "
            "This usually means .X is corrupted or not a numeric count matrix."
        )
        return

    total_elements = n_rows * n_cols

    try:
        if issparse(matrix):
            # SciPy sparse matrix
            non_zero_elements = int(matrix.nnz)
        else:
            # In backed mode or dense: convert to numpy array safely
            arr = np.asarray(matrix)

            if arr.ndim != 2:
                print(
                    f"\nMatrix density calculation skipped: '{name}' became {arr.ndim}D after np.asarray."
                )
                return

            if not np.issubdtype(arr.dtype, np.number):
                print(
                    f"\nMatrix density calculation skipped: '{name}' array dtype {arr.dtype} is non-numeric."
                )
                return

            non_zero_elements = int(np.count_nonzero(arr))

        percentage_non_zero = (non_zero_elements / total_elements) * 100.0

        print(f"\n--- Matrix Density for {name} ---")
        print(f"Non-zero elements: {non_zero_elements:,}")
        print(f"Total elements: {total_elements:,}")
        print(f"Percentage of Non-Zero Values (Density): {percentage_non_zero:.4f}%")

    except Exception as e:
        print(f"\nError calculating density for '{name}': {e}")


def calculate_nonzero_cells(matrix, name):
    """
    Safely calculates how many rows (cells) have at least one non-zero entry.
    """
    if matrix is None:
        print(f"\nPer-cell non-zero calculation skipped: '{name}' is None.")
        return

    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        print(f"\nPer-cell non-zero calculation skipped: '{name}' has invalid shape {shape}.")
        return

    n_rows, n_cols = shape
    if n_rows == 0 or n_cols == 0:
        print(f"\nPer-cell non-zero calculation skipped: '{name}' has zero rows or columns.")
        return

    dtype = getattr(matrix, "dtype", None)
    if dtype is not None and not np.issubdtype(dtype, np.number):
        print(
            f"\nPer-cell non-zero calculation skipped: '{name}' has non-numeric dtype {dtype}. "
            "This usually means .X is corrupted or not numeric."
        )
        return

    try:
        if issparse(matrix):
            row_nnz = matrix.getnnz(axis=1)
        else:
            arr = np.asarray(matrix)
            if arr.ndim != 2:
                print(
                    f"\nPer-cell non-zero calculation skipped: '{name}' became {arr.ndim}D after np.asarray."
                )
                return

            if not np.issubdtype(arr.dtype, np.number):
                print(
                    f"\nPer-cell non-zero calculation skipped: '{name}' array dtype {arr.dtype} is non-numeric."
                )
                return

            row_nnz = np.count_nonzero(arr, axis=1)

        row_nnz = np.asarray(row_nnz).ravel()
        cells_with_signal = int((row_nnz > 0).sum())
        total_cells = n_rows
        perc_cells = cells_with_signal / total_cells * 100.0

        print(f"\n--- Cells with non-zero values in {name} ---")
        print(
            f"{cells_with_signal} of {total_cells} cells "
            f"({perc_cells:.4f}%) have at least one non-zero entry."
        )

    except Exception as e:
        print(f"\nError calculating per-cell non-zero counts for '{name}': {e}")


def inspect_anndata_file(filepath):
    """
    Loads an H5AD file and prints metadata + safe summaries of .X and .obs.
    """
    file_path = Path(filepath)

    if not file_path.exists():
        print(f"ERROR: File not found at path: {filepath}")
        return

    try:
        print(f"Loading AnnData object from: {file_path.name}")
        # Use normal (in-memory) mode; safer for inspection
        adata = anndata.read_h5ad(file_path)

        print("\n--- Summary ---")
        print(f"Dimensions (Cells x Genes): {adata.n_obs} x {adata.n_vars}")

        if not adata.obs_names.empty:
            print("\n--- First 5 Cell Names (adata.obs_names) ---")
            for i, name in enumerate(adata.obs_names[:5]):
                print(f"  {i+1}. {name}")
        else:
            print("\n--- Cell Names Not Found ---")

        print("\n--- All Columns in AnnData.obs (Cell Metadata) ---")
        print(adata.obs.columns.tolist())

        # Main matrix stats
        calculate_density(adata.X, "adata.X (Main Matrix)")
        calculate_nonzero_cells(adata.X, "adata.X (Main Matrix)")

        # --- Unique value summaries for selected obs columns ---
        obs_cols = [
            "Sample_ID",
            "Cancer_Type",
            "Cancer_Status",
            "Response_Status",
            "Condition",
            "cell_type",
            "Chi",
        ]

        for oc in obs_cols:
            if oc in adata.obs.columns:
                series = adata.obs[oc]
                # Use pandas.unique (handles mixed types / NaNs safely, no '<' comparisons)
                unique_v = pd.unique(series)
                num_unique = len(unique_v)
                print(f"\n--- Unique values in adata.obs['{oc}'] ({num_unique} total) ---")
                if num_unique <= 50:
                    print(unique_v)
                else:
                    print(unique_v[:50])

        print(f"\n--- Unique Check ---")
        if "Sample_ID" in adata.obs.columns and "Cancer_Status" in adata.obs.columns:
            try:
                mask = adata.obs["Sample_ID"].isin(["C1", "C2", "C3"])
                print(adata.obs.loc[mask, "Cancer_Status"].unique())
            except Exception as e:
                print(f"(Unique Check skipped due to error: {e})")
        else:
            print("Sample_ID or Cancer_Status not found; skipping Unique Check.")

    except Exception as e:
        print(f"An unexpected error occurred while processing the file: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_h5ad.py <path_to_h5ad_file>")
        sys.exit(1)

    h5ad_filepath = sys.argv[1]

    inspect_anndata_file(h5ad_filepath)
