import scanpy as sc
import os
import pandas as pd
import numpy as np # Needed for creating the new temporary color column
import matplotlib as mpl # NEW: Added for robust access to the default color cycle

# New function to handle specific subset plotting
def plot_subset_umap(adata: sc.AnnData, output_dir: str):
    """
    Creates a specific UMAP plot for a subset of cells based on user-defined criteria,
    showing all other cells as a background color.
    """
    print("\n=====================================================")
    print("Starting Specific Subset UMAP Plotting (with Background)")
    print("=====================================================")

    # --- 0. Define Filter Criteria ---
    GROUP_KEY = 'Sample_ID'
    group_values_to_keep = {'A1', 'A10', 'A11', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B1', 'B10', 'B11'
                            , 'B12', 'B13', 'B14', 'B14B', 'B15', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9'
                            , 'HCC1143', 'MCF7', 'SUM149PT', 'T47D''C1','MM468-5FU1-D33-H3K27me3','MM468-5FU1-D60-H3K4me3'
                            , 'MM468-5FU2-D171-H3K27me3','MM468-5FU2-D67-H3K27me3','MM468-5FU3-D147-H3K27me3','MM468-5FU6-D131-H3K27me3'
                            , 'MM468-GSK4-D91-H3K27me3','MM468-Untreated-D0-H3K4me3','MM468-Untreated-D131-H3K27me3'
                            ,'MM468-Untreated-D60-H3K27me3','MM468-Untreated-D77-H3K27me3'}
    SUBGROUP_KEY = 'Response_Status'
    subgroup_values_to_keep = ['TNBC','Resistant', 'Equivocal', 'Her2neg', 'Unknown']
    BACKGROUND_COLOR_LABEL = 'Other Cells'
    subset_title= "UMAP Subset: Epithelial and Stromal Cells vs. Background"

    # # # --- 1. Define Filter Criteria ---
    # GROUP_KEY = 'Batch_ID'
    # SUBGROUP_KEY = 'cell_type'
    # group_values_to_keep = {'A', 'B', 'C', 'CellLine'}
    # # subgroup_values_to_keep = ['CD8_T', 'CD4_T', 'B_Cells','Plasma', 'Monocytes', 'Macrophages', 'Mast']
    # subgroup_values_to_keep = ['BC', 'OC', 'EC', 'GC', 'Normal_Epithelial', 'Endothelial', 'Fibroblasts', 'Pericytes', 'Myofibroblasts']
    # # Define the background color label (will be mapped to a color later)
    # BACKGROUND_COLOR_LABEL = 'Other Cells'
    # # subset_title= "UMAP Subset: Immune Cells vs. Background)"
    # subset_title= "UMAP Subset: Epithelial and Stromal Cells vs. Background"
    
    # # # --- 1.1 Define Filter Criteria ---
    # GROUP_KEY = 'cell_type'
    # SUBGROUP_KEY = 'Response_Status'
    # group_values_to_keep = {'BC'}
    # # subgroup_values_to_keep = ['CD8_T', 'CD4_T', 'B_Cells','Plasma', 'Monocytes', 'Macrophages', 'Mast']
    # subgroup_values_to_keep = ['Chemonaive', 'Persister', 'Residual', 'Resistant', 'Recurrent']
    # # Define the background color label (will be mapped to a color later)
    # BACKGROUND_COLOR_LABEL = 'Other Cells'
    # # subset_title= "UMAP Subset: Immune Cells vs. Background)"
    # subset_title= "UMAP Subset: BC in Different Response Status"
    
    if GROUP_KEY not in adata.obs or SUBGROUP_KEY not in adata.obs:
        print(f"Error: One or both required keys ('{GROUP_KEY}' or '{SUBGROUP_KEY}') not found in adata.obs.")
        return

    # --- 2. Create Boolean Index for Subsetting ---
    batch_filter = adata.obs[GROUP_KEY].isin(group_values_to_keep)
    cancer_filter = adata.obs[SUBGROUP_KEY].isin(subgroup_values_to_keep)
    combined_filter = batch_filter & cancer_filter

    n_subset_cells = combined_filter.sum()
    
    # Note: The traceback showed n_subset_cells > 0, so we continue even if the number seems small compared to the total.
    if n_subset_cells == 0:
        print("Warning: Combined filter resulted in zero cells. Skipping subset plot.")
        return

    print(f"Subset identified: {n_subset_cells} cells selected out of {adata.n_obs}.")

    # --- 3. Create a Temporary Color Column in the FULL adata object ---
    TEMP_COLOR_KEY = 'temp_subset_color'
    
    # Initialize the new column with the background label for ALL cells
    adata.obs[TEMP_COLOR_KEY] = BACKGROUND_COLOR_LABEL
    
    # Assign the SUBGROUP_KEY value (e.g., 'CD8_T', 'B_Cells', etc.) to the subset cells
    adata.obs.loc[combined_filter, TEMP_COLOR_KEY] = adata.obs.loc[combined_filter, SUBGROUP_KEY]
    
    ordered_categories = [BACKGROUND_COLOR_LABEL] + subgroup_values_to_keep
    
    # 3. Filter the ordered list to only include categories actually present in the data
    # (This step is good practice, though not strictly necessary if all are present)
    present_categories = [cat for cat in ordered_categories if cat in adata.obs[TEMP_COLOR_KEY].unique()]
    
    adata.obs[TEMP_COLOR_KEY] = adata.obs[TEMP_COLOR_KEY].astype(
        pd.CategoricalDtype(categories=present_categories)
    )

    # --- 4. Define and Apply Custom Colormap ---
    # We use a custom color map to explicitly set the background color
    
    # FIX: Replaced 'sc.plotting.palettes.default_cycler' which caused AttributeError
    # Access the default color cycle using the imported matplotlib configuration (rcParams)
    default_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    VIBRANT_PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', #'#1f77b4' '#9ecae1'
        '#e377c2', '#bcbd22', '#17becf', '#f7b6d2'             
    ]
    # Create the mapping for the custom colors
    custom_colors = {}
    alpha_value_for_background = 0.1
    
    for i, category in enumerate(ordered_categories):
        if category == BACKGROUND_COLOR_LABEL:
            custom_colors[category] = 'lightgray' # The explicit background color
            custom_colors[category] = (211/255, 211/255, 211/255, alpha_value_for_background)
        else:
            # Cycle through the default colors for the subset categories
            # We offset the index by 1 because the first category is the background color
            # The colors for the actual subset groups should start from the 0th default color.
            custom_colors[category] = VIBRANT_PALETTE[(i - 1) % len(default_colors)]
            
    # Map the categories to the color list in the correct order
    color_list = [custom_colors[cat] for cat in ordered_categories]


    # --- 5. Plot the Full UMAP (colored by the Temporary Key) ---
    plot_key = TEMP_COLOR_KEY
    filename_suffix = f'{SUBGROUP_KEY}_in_{GROUP_KEY}_vs_Background.png'

    print(f"Plotting UMAP with subset colored by: {SUBGROUP_KEY} vs. Background")

    sc.settings.figdir = output_dir  # ensure Scanpy saves here
    sc.pl.umap(
        adata, 
        color=plot_key, 
        title=subset_title,
        save=filename_suffix, 
        show=False,
        frameon=False,
        palette=color_list,
        size=2
    )
    print(f"  -> ✓ Saved: {os.path.join(output_dir, 'umap' + filename_suffix)}")


    # Clean up the temporary column
    del adata.obs[TEMP_COLOR_KEY]
    
# The plot_umap_from_h5ad and __main__ blocks remain the same for calling the function
def plot_umap_from_h5ad(h5ad_path: str, batch_key: str = 'Batch_ID', cluster_key: str = 'leiden', output_dir: str = 'umap_plots'):
    """Loads AnnData and generates UMAP plots, including standard and subset-highlight plots."""
    print("=====================================================")
    print(f"Loading AnnData from: {h5ad_path}")
    print("=====================================================")

    try:
        # 1. Load the AnnData object
        adata = sc.read_h5ad(h5ad_path)
    except FileNotFoundError:
        print(f"Error: File not found at {h5ad_path}. Please check the path and try again.")
        return

    # Check for UMAP coordinates, which are essential for plotting
    if 'X_umap' not in adata.obsm:
        print("Error: UMAP coordinates ('X_umap') not found in adata.obsm.")
        print("Please ensure the UMAP step was completed before saving the .h5ad file.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to the '{output_dir}' directory...")
    
    # Temporarily change the working directory for scanpy to save plots directly there
    original_cwd = os.getcwd()
    # Check if we are already in the target directory (prevents issues if calling from the target dir)
    if os.path.abspath(original_cwd) != os.path.abspath(output_dir):
        sc.settings.figdir = '.' # Tell scanpy to save figures relative to the CWD
        os.chdir(output_dir)
    else:
        sc.settings.figdir = '.'


    # --- Plot 1: UMAP colored by Clustering (leiden) ---
    if cluster_key in adata.obs:
        print(f"\nPlotting UMAP colored by: {cluster_key}")
        sc.pl.umap(
            adata, 
            color=cluster_key, 
            title=f"UMAP colored by {cluster_key}", 
            save=f'_{cluster_key}.png', # scanpy auto-prepends 'umap'
            show=False,
            frameon=False,
        )
        print(f"  -> ✓ Saved: umap_{cluster_key}.png")
    else:
        print(f"Warning: Cluster key '{cluster_key}' not found in adata.obs. Skipping cluster plot.")
    
    # --- Batch Label Remapping and Plotting ---
    label_map = {
        'SampleA': 'Condition_1_Control',
        'SampleB': 'Condition_1_Treated',
        'SampleC': 'Condition_2_Control',
        'SampleD': 'Condition_2_Treated'
    }

    batch_key_column = batch_key
    RENAMED_KEY = f'{batch_key_column}_renamed' # Define the renamed key here
    
    if batch_key_column in adata.obs:
        adata.obs[RENAMED_KEY] = adata.obs[batch_key_column].apply(
             lambda x: label_map[x] if x in label_map else x
        )
        
        # --- Plot 2a: UMAP colored by New/Renamed Labels ---
        print(f"\nPlotting UMAP colored by: {RENAMED_KEY} (Renamed if mapped)")
        sc.pl.umap(
            adata,
            color=RENAMED_KEY,
            title=f"UMAP colored by {batch_key_column} (Renamed if mapped)",
            save=f'_{RENAMED_KEY}.png',
            show=False,
            frameon=False
        )
        print(f"  -> ✓ Saved: umap_{RENAMED_KEY}.png")
    else:
        print(f"Warning: Batch key '{batch_key}' not found in adata.obs. Skipping batch plots.")

    # --- Plot 3: Specific Subset Plot ---
    # This calls the new function
    plot_subset_umap(adata, output_dir)
        
    # Restore the original working directory only if we changed it
    if os.path.abspath(original_cwd) != os.path.abspath(output_dir):
        os.chdir(original_cwd)
    
    print("\n✅ Plotting complete and working directory restored.")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # --- CONFIGURATION ---
    # ------------------------------------------------------------------
    sc.settings.set_figure_params(dpi_save=300)
    # 1. Update this path to your actual .h5ad file location
    # OUTPUT_DIR = '/mnt/f/H3K27/Data/scRNA/'
    # H5AD_PATH = OUTPUT_DIR+'scRNA_annotated.h5ad'
    
    OUTPUT_DIR = '/mnt/f/H3K27/Data/scATAC/'
    H5AD_PATH = OUTPUT_DIR+'scATAC_accessibility_processed.h5ad'
    
    # OUTPUT_DIR = '/mnt/f/H3K27/Data/scChi/'
    # H5AD_PATH = OUTPUT_DIR+'scChi_accessibility_processed.h5ad'

    # 3. Define the key you want for the batch check plot
    BATCH_KEY_USER = 'Batch_ID' # <-- Change this to the actual column name for your batches
    
    # 4. Define the key for the clustering result
    CLUSTER_KEY_DEFAULT = 'leiden'

    plot_umap_from_h5ad(
        h5ad_path=H5AD_PATH,
        batch_key=BATCH_KEY_USER,
        cluster_key=CLUSTER_KEY_DEFAULT,
        output_dir=OUTPUT_DIR
    )
