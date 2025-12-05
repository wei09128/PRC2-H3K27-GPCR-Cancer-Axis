import os
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import warnings
from scipy.stats import median_abs_deviation

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore', category=FutureWarning)

def filter_multiplets_by_counts(adata: ad.AnnData, count_key: str, mad_threshold: int = 5) -> ad.AnnData:
    """
    Filters potential multiplets (doublets) from an scATAC-seq or scChIP-seq AnnData object 
    based on the number of unique fragments/counts per cell using the MAD method.

    This function removes cells in the extreme upper tail of the fragment count distribution, 
    where cells containing two nuclei typically reside.

    Args:
        adata: The single-cell epigenomic AnnData object.
        count_key: The column name in adata.obs containing the fragment/UMI counts 
                   (e.g., 'n_fragments' or 'total_counts').
        mad_threshold: The number of Median Absolute Deviations (MADs) above the median 
                       to set as the upper count limit for singlets. Default is 5.

    Returns:
        The filtered AnnData object containing only predicted singlets.
    """
    
    print(f"\n[Filtering Module] Original cell count: {adata.n_obs}")
    
    # Check for the required count column
    if count_key not in adata.obs.columns:
        print(f"Error: Count key '{count_key}' not found in adata.obs.")
        print(f"Available keys: {list(adata.obs.columns)}")
        # If the key is missing, return the original data and log the error
        return adata
        
    counts = adata.obs[count_key].values

    # 1. Calculate robust statistics (Median and MAD)
    median_counts = np.median(counts)
    
    # Calculate MAD (Median Absolute Deviation) - Fixed: removed scale parameter
    mad = median_abs_deviation(counts)

    # Handle case where MAD is zero (e.g., highly uniform, unlikely in sc data)
    if mad == 0:
        print("Warning: MAD is zero. Falling back to 3 standard deviations.")
        std_dev = np.std(counts)
        upper_threshold = median_counts + (mad_threshold * std_dev)
    else:
        # 2. Define the upper threshold: Median + (MAD_Threshold * MAD)
        upper_threshold = median_counts + (mad_threshold * mad)
    
    # Log the metrics
    print(f"  > Metric: {count_key}")
    print(f"  > Median unique fragments: {median_counts:.0f}")
    print(f"  > MAD: {mad:.0f}")
    print(f"  > Upper Threshold (Median + {mad_threshold}*MAD): {upper_threshold:.0f}")

    # 3. Filter the AnnData object to keep only cells below the threshold
    singlet_mask = adata.obs[count_key] < upper_threshold
    
    # Store the multiplet flag before filtering
    adata.obs['is_multiplet'] = ~singlet_mask
    
    adata_singlets = adata[singlet_mask].copy()
    
    doublets_removed = adata.n_obs - adata_singlets.n_obs
    removal_percentage = (doublets_removed / adata.n_obs) * 100

    print(f"  > Cells identified as multiplets: {doublets_removed} ({removal_percentage:.2f}%)")
    print(f"  > Final cell count after filtering: {adata_singlets.n_obs}")

    return adata_singlets


def batch_process_doublets(input_folder: str, output_folder: str, count_key: str, mad_threshold: int):
    """
    Iterates through all H5AD files in the input folder, applies doublet filtering, 
    and saves the filtered result to the output folder.
    """
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Find all .h5ad files in the input folder
    h5ad_files = [f for f in os.listdir(input_folder) if f.endswith('.h5ad')]
    
    if not h5ad_files:
        print(f"Error: No .h5ad files found in the input folder: {input_folder}")
        return

    print(f"\n{'='*70}")
    print(f"STARTING BATCH PROCESSING: Found {len(h5ad_files)} files to process.")
    print(f"Filtering Parameter: {count_key} < Median + {mad_threshold}*MAD")
    print(f"{'='*70}")

    # Process each file one by one
    for filename in sorted(h5ad_files):
        input_file_path = os.path.join(input_folder, filename)
        
        print(f"\n>>> Processing file: {filename} <<<")
        
        try:
            # 1. Load the AnnData object
            adata = sc.read_h5ad(input_file_path)
            
            # 2. Perform Multiplet Filtering
            adata_singlets = filter_multiplets_by_counts(adata, count_key, mad_threshold)
            
            # 3. Save the filtered file
            if adata_singlets.n_obs < adata.n_obs:
                suffix = "_singlets"
            else:
                suffix = "_unfiltered" # If no doublets were removed
            
            
            output_filename = filename.replace(".h5ad", "_singlets.h5ad")
            output_file_path = os.path.join(output_folder, output_filename)
            
            adata_singlets.write(output_file_path, compression='gzip')
            print(f"  > Saved filtered file to: {output_file_path}")

        except Exception as e:
            # Enhanced error reporting to show the exception details clearly
            print(f"\n[CRITICAL ERROR] Failed to process {filename}: {e}")
            import traceback
            print(traceback.format_exc())
            print("Skipping this file and continuing...")
        
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE.")
    print(f"{'='*70}")


# ============================================================================
# --- CONFIGURATION ---
# ============================================================================

# !!! USER INPUT REQUIRED !!!
# 1. Path to the folder containing your scATAC/scChIP H5AD files.
INPUT_DIR = "/mnt/f/H3K27/GSE164716_scATAC/"

# 2. Path to the folder where the filtered H5AD files will be saved.
OUTPUT_DIR = "/mnt/f/H3K27/scChi_clean/"

# 3. The exact column name in adata.obs containing the unique fragment counts 
#    Common options:
#    - 'total_counts' (most common for scATAC-seq)
#    - 'n_fragment' or 'n_fragments'
#    - 'passed_filters' (Cell Ranger ATAC)
FRAGMENT_COUNT_KEY = "total_counts" 

# 4. The statistical threshold for filtering (5 is a robust default).
#    A lower number (e.g., 3-4) is more aggressive (removes more cells).
#    A higher number (e.g., 6-7) is more conservative (removes fewer cells).
MAD_THRESHOLD = 5

# ============================================================================
# --- EXECUTION ---
# ============================================================================

if __name__ == "__main__":
    batch_process_doublets(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        count_key=FRAGMENT_COUNT_KEY,
        mad_threshold=MAD_THRESHOLD
    )