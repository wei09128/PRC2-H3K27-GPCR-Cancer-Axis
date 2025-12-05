import anndata as ad
import os
import glob
import re
import pandas as pd
import sys

def standardize_cell_names(directory_path, keep_columns):
    """
    Loads all .h5ad files, fixes non-unique var names, standardizes cell names 
    by ensuring format is 'prefix_barcode':
    - If exactly one '_': keep as is
    - If no '_': add 'prefix_' 
    - If more than one '_': keep everything, treating the part before the last '_' as prefix
    Keeps only specified columns in .obs, and saves the modified file.
    """
    h5ad_files = glob.glob(os.path.join(directory_path, '*.h5ad'))
    
    if not h5ad_files:
        print(f"Error: No .h5ad files found in the directory: {directory_path}", file=sys.stderr)
        return

    print("--- Starting Cell and Column Standardization ---")
    print(f"Target columns to keep in .obs: {', '.join(keep_columns)}")
    print("-" * 60)
    
    for filename in h5ad_files:
        basename = os.path.basename(filename)
        print(f"Processing file: {basename}")
        
        try:
            # 1. Load the AnnData object completely
            adata = ad.read_h5ad(filename)
            
            # 2. 🌟 FIX NON-UNIQUE VARIABLE NAMES 🌟
            adata.var_names_make_unique()
            
            # 3. ⭐️ DETERMINE SAMPLE PREFIX FROM FILENAME ⭐️
            # Strip the file extension and split by underscore
            parts = basename.replace('.h5ad', '').split('_')
            
            if len(parts) < 2:
                print(f"  > WARNING: Filename structure is unusual, skipping cell name prefixing.")
                prefix = None
            else:
                # Always use the 2nd element (index 1)
                prefix = parts[1]
            
            # 4. ⭐️ STANDARDIZE CELL NAMES (.obs_names) ⭐️
            if prefix:
                new_obs_names = []
                modified = False
                
                for name in adata.obs_names:
                    underscore_count = name.count('_')
                    
                    if underscore_count == 1:
                        # Has exactly one underscore, keep as is
                        new_obs_names.append(name)
                    elif underscore_count == 0:
                        # No underscore, add prefix with underscore
                        new_name = f'{prefix}_{name}'
                        new_obs_names.append(new_name)
                        modified = True
                    else:
                        # More than one underscore, keep as is (already has prefix_barcode format)
                        # E.g., 'MM468-Untreated-D0-H3K4me3_ATTCGTTCATGGTATC-1' stays the same
                        new_obs_names.append(name)
                
                if modified:
                    adata.obs_names = new_obs_names
                    print(f"  > Cell names standardized with prefix: {prefix}_")
                else:
                    print(f"  > Cell names already in correct format. Skipping.")
            
            # 5. SUBSET THE .obs DATAFRAME (QC Standardization)
            
            # Initialize missing columns before subsetting
            missing_columns = [col for col in keep_columns if col not in adata.obs.columns]
            if missing_columns:
                print(f"  > WARNING: Missing required .obs columns: {', '.join(missing_columns)}. Initializing.")
                for col in missing_columns:
                    if col == 'passed_qc':
                        adata.obs[col] = False
                    else:
                        adata.obs[col] = pd.NA
            
            # Subset the .obs to only include the target columns
            adata.obs = adata.obs[keep_columns]

            print(f"  > Retained .obs columns: {', '.join(keep_columns)}")
            
            # 6. Save the modified AnnData object (no backup)
            adata.write(filename, compression='gzip')
            print(f"  > Successfully updated and saved: {basename}")
            
        except Exception as e:
            print(f"  > FATAL ERROR: Could not process {basename}. Check file integrity. Error: {e}", file=sys.stderr)
            continue
    
    print("\n--- Standardization COMPLETE ---")
            
# --- Configuration and Execution ---

TARGET_COLUMNS = [
    'n_genes_by_counts',
    'total_counts',
    'total_counts_mt',
    'pct_counts_mt',
    'passed_qc',
    'n_genes',
    'Batch_ID',
    'Sample_ID',
    'Cancer_Type',
    'Cancer_Status',
    'Response_Status'
]

# Run the standardization on the current directory
current_directory = '.'
standardize_cell_names(current_directory, TARGET_COLUMNS)