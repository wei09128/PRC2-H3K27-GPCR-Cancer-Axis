import scanpy as sc
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import argparse
import sys
import matplotlib.pyplot as plt

# --- Helper Functions (Core Logic) ---

def parse_comma_separated_list(arg_list: str) -> set:
    """Safely parses a comma-separated string into a set of strings."""
    if not arg_list:
        return set()
    # Strip whitespace and split by comma
    return set(x.strip() for x in arg_list.split(','))

def plot_subset_umap(
    adata: sc.AnnData, 
    output_dir: str, 
    group_key: str, 
    group_values: set, 
    subgroup_key: str, 
    subgroup_values: set,
    background_label: str, 
    title: str
):
    """
    Creates a specific UMAP plot for a subset of cells based on user-defined criteria,
    showing all other cells as a transparent background color.
    """
    print("\n=====================================================")
    print("Starting Specific Subset UMAP Plotting (with Background)")
    print("=====================================================")

    # --- 0. Define Filter Criteria ---
    if group_key not in adata.obs or subgroup_key not in adata.obs:
        print(f"Error: One or both required keys ('{group_key}' or '{subgroup_key}') not found in adata.obs. Skipping plot.")
        return

    # --- 1. Create Boolean Index for Subsetting ---
    batch_filter = adata.obs[group_key].isin(group_values)
    cancer_filter = adata.obs[subgroup_key].isin(subgroup_values)
    combined_filter = batch_filter & cancer_filter

    n_subset_cells = combined_filter.sum()
    
    if n_subset_cells == 0:
        print("Warning: Combined filter resulted in zero cells. Skipping subset plot.")
        return

    print(f"Subset identified: {n_subset_cells} cells selected out of {adata.n_obs}.")
    print(f"Filter: {group_key} in {group_values} AND {subgroup_key} in {subgroup_values}")

    # --- 2. Create a Temporary Color Column in the FULL adata object ---
    TEMP_COLOR_KEY = 'temp_subset_color'
    adata.obs[TEMP_COLOR_KEY] = background_label
    adata.obs.loc[combined_filter, TEMP_COLOR_KEY] = adata.obs.loc[combined_filter, subgroup_key]
    
    subset_groups_list = sorted(list(subgroup_values))
    ordered_categories = [background_label] + subset_groups_list
    present_categories = [cat for cat in ordered_categories if cat in adata.obs[TEMP_COLOR_KEY].unique()]
    
    # 🌟 FIX: Convert to categorical and THEN explicitly reorder to ensure plotting layer works
    adata.obs[TEMP_COLOR_KEY] = adata.obs[TEMP_COLOR_KEY].astype('category')
    adata.obs[TEMP_COLOR_KEY] = adata.obs[TEMP_COLOR_KEY].cat.reorder_categories(present_categories, ordered=True)

    # --- 3. Define and Apply Custom Colormap with Transparency ---
    VIBRANT_PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#e377c2', '#bcbd22', '#17becf', '#f7b6d2' 
    ]
        # '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        # '#e377c2', '#bcbd22', '#17becf', '#f7b6d2' 
    custom_colors = {}
    alpha_value_for_background = 0.005 # Drastically low alpha to prevent saturation
    size_for_background = 2           # Very small size to minimize stacking
    size_for_subset = 5              # Keep subset visible
     
    # Determine the index offset for cycling through the vibrant palette
    subset_categories = [cat for cat in present_categories if cat != background_label]
     
    # The background color is an RGBA tuple
    background_rgba = (0.827, 0.827, 0.827, alpha_value_for_background) # Light Gray
    
    # Build the palette list for the full plotting approach
    color_list_full = []
    for category in present_categories:
        if category == background_label:
            color_list_full.append(background_rgba)
        else:
            subset_idx = subset_categories.index(category)
            # For simplicity, convert HEX subset colors to opaque RGBA to match format
            hex_color = VIBRANT_PALETTE[subset_idx % len(VIBRANT_PALETTE)]
            # mpl.colors.to_rgba can convert HEX to RGBA
            rgb = mpl.colors.to_rgb(hex_color)
            color_list_full.append(rgb + (1.0,)) # Append with alpha=1.0 (opaque)
    
    # Verify the list length
    print(f"Palette length: {len(color_list_full)}, Categories length: {len(present_categories)}")
    
    # --- 4. Plot the UMAP (Multi-Layer Z-Order) ---
    filename_suffix = f'{subgroup_key}_in_{group_key}_vs_Background_subset.png'
    print(f"Plotting UMAP with subset colored by: {subgroup_key} vs. Background (Transparent)")
    
    # --- 4.1. Setup Figure and Axes ---
    # Initialize the figure and axis explicitly (Standard Matplotlib way)
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title(title) # Set the title on the Matplotlib axis
    
    # --- 4.2. Plot 1: Transparent Background Layer (Z-order=0) ---
    # Filter AnnData to ONLY the background cells
    combined_filter = adata.obs[group_key].isin(group_values) & adata.obs[subgroup_key].isin(subgroup_values)
    adata_background = adata[~combined_filter, :].copy()
    
    # Filter color_list_full to only include the transparent background color
    color_list_background = [c for c in color_list_full if c == background_rgba]
    
    sc.pl.umap(
        adata_background,          # Subset AnnData for Background ONLY
        color=TEMP_COLOR_KEY,
        ax=ax,                     # Plot onto the established axis
        show=False,
        title='UMAP '+title,      # Title already set
        frameon=False,
        palette=color_list_background, # Only the transparent color
        size=size_for_background,  # Use the small size
        zorder=0,                  # Plot underneath
        legend_loc=None,           # No legend for background
        save=False,
    )
    
    # --- 4.3. Plot 2: Opaque Subset Layer (Z-order=1) ---
    # Filter AnnData to ONLY the subset cells
    adata_subset = adata[combined_filter, :].copy()
    
    # Filter color_list_full to only include the opaque subset colors (excluding background)
    color_list_subset = [c for c in color_list_full if c != background_rgba]
    
    sc.pl.umap(
        adata_subset,              # Subset AnnData for Foreground ONLY
        color=TEMP_COLOR_KEY,
        ax=ax,                     # Plot onto the established axis (The fix!)
        show=False,
        title='UMAP Subset: '+title,
        frameon=False,
        palette=color_list_subset, # Only the vibrant colors
        size=size_for_subset,      # Use the large size
        zorder=1,                  # Plot on top
        legend_loc='on data',
        legend_fontsize=10,
        save=False
    )
    
    # --- 4.4. Saving the Combined Figure ---
    # The figure object is 'fig' from plt.subplots()
    fig.savefig(os.path.join(sc.settings.figdir, title+'_layered.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f" -> ✓ Saved layered plot: {title+'_layered.png'}")
    
    # --- 4.5. Save the Separate Legend Plot (Right Margin) ---
    sc.pl.umap(
        adata_subset, 
        color=TEMP_COLOR_KEY, 
        title='UMAP Subset: '+title,
        save='UMAP Subset: '+title+'_legend.png', 
        show=False,
        frameon=False,
        palette=color_list_subset,
        size=size_for_subset,
        legend_fontsize=10,
        legend_loc='right margin',
    )
    print(f" -> ✓ Saved legend plot: {title+'_legend.png'}")
    
    # Clean up the temporary column
    del adata.obs[TEMP_COLOR_KEY]
    adata.obs = adata.obs.copy()

def plot_umap_from_h5ad(
    h5ad_path: str, 
    output_dir: str,
    batch_key: str, 
    cluster_key: str,
    label_map: dict,
    group_key: str,
    group_values: set,
    subgroup_key: str,
    subgroup_values: set,
    background_label: str,
    subset_title: str
):
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

    if 'X_umap' not in adata.obsm:
        print("Error: UMAP coordinates ('X_umap') not found in adata.obsm. Skipping all plots.")
        return

    # Ensure output directory exists and set scanpy figdir
    os.makedirs(output_dir, exist_ok=True)
    sc.settings.figdir = output_dir 
    print(f"Saving plots to the '{output_dir}' directory...")

    # --- Plot 1: UMAP colored by Clustering (leiden) ---
    if cluster_key in adata.obs:
        print(f"\nPlotting UMAP colored by: {cluster_key}")
        sc.pl.umap(
            adata, 
            color=cluster_key, 
            title=f"UMAP colored by {cluster_key}", 
            save=f'_{cluster_key}.png',
            show=False,
            frameon=False,
            size=2,
            legend_loc='right margin',      # <- ADDED (often best for clusters)
        )
        print(f"  -> ✓ Saved: umap_{cluster_key}.png")
    else:
        print(f"Warning: Cluster key '{cluster_key}' not found in adata.obs. Skipping cluster plot.")
    
    # --- Batch Label Remapping and Plotting ---
    RENAMED_KEY = f'{batch_key}_renamed' 
    
    if batch_key in adata.obs:
        print(f"\nProcessing batch key '{batch_key}' with {len(label_map)} renames...")
        adata.obs[RENAMED_KEY] = adata.obs[batch_key].apply(
             lambda x: label_map.get(x, x) # Use .get() for safe mapping
        )
        
        # --- Plot 2a: UMAP colored by New/Renamed Labels ---
        print(f"Plotting UMAP colored by: {RENAMED_KEY} ")
        sc.pl.umap(
            adata,
            color=RENAMED_KEY,
            title=f"UMAP colored by Cohort Type",
            save=f'_{RENAMED_KEY}.png',
            show=False,
            frameon=False,
            size=2
        )
        print(f"  -> ✓ Saved: umap_{RENAMED_KEY}.png")
    else:
        print(f"Warning: Batch key '{batch_key}' not found in adata.obs. Skipping batch plots.")

    # --- Plot 3: Specific Subset Plot ---
    # This calls the parameterized function
    plot_subset_umap(
        adata, 
        output_dir,
        group_key, 
        group_values, 
        subgroup_key, 
        subgroup_values,
        background_label, 
        subset_title
    )
    
    print("\n✅ Plotting complete.")


def main():
    """Parses command line arguments and runs the plotting pipeline."""
    sc.settings.set_figure_params(dpi_save=300)
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['legend.fontsize'] = 8      # Smaller legend text
    plt.rcParams['axes.labelsize'] = 10      # Smaller axis labels (UMAP 1, UMAP 2)
    plt.rcParams['axes.titlesize'] = 12      # Smaller plot title
    
    parser = argparse.ArgumentParser(
        description="Generate UMAP plots with specific subset highlighting from a processed AnnData file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- File/Path Arguments ---
    parser.add_argument(
        '--h5ad_path',
        type=str,
        required=True,
        help="Path to the input processed AnnData file (e.g., scChi_accessibility_processed.h5ad)."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory to save all UMAP plots."
    )
    
    # --- Standard Plotting Arguments ---
    parser.add_argument(
        '--batch_key_user',
        type=str,
        default='Batch_ID',
        help="The column name in adata.obs to use for the batch color plot (Default: Batch_ID)."
    )
    parser.add_argument(
        '--cluster_key',
        type=str,
        default='leiden',
        help="The column name for clustering results (Default: leiden)."
    )

    # --- Subset Plotting Arguments ---
    parser.add_argument(
        '--group_key',
        type=str,
        default='Sample_ID',
        help="The main column key for the primary filter (e.g., Sample_ID)."
    )
    parser.add_argument(
        '--group_values_to_keep',
        type=str,
        required=True,
        help="Comma-separated list of values in GROUP_KEY to include in the subset (e.g., A1,A2,B1,B2)."
    )
    parser.add_argument(
        '--subgroup_key',
        type=str,
        default='Response_Status',
        help="The secondary column key to color the subset by (e.g., Response_Status)."
    )
    parser.add_argument(
        '--subgroup_values_to_keep',
        type=str,
        required=True,
        help="Comma-separated list of values in SUBGROUP_KEY to include in the subset (e.g., Resistant,Equivocal,TNBC)."
    )
    parser.add_argument(
        '--background_color_label',
        type=str,
        default='Other Cells',
        help="The categorical label used for cells not in the subset (Default: Other Cells)."
    )
    parser.add_argument(
        '--subset_title',
        type=str,
        default="UMAP Subset: Highlighted Cells vs. Background",
        help="Title for the subset UMAP plot."
    )

    args = parser.parse_args()

    # --- Complex Parameter Handling ---
    group_values_to_keep_set = parse_comma_separated_list(args.group_values_to_keep)
    subgroup_values_to_keep_set = parse_comma_separated_list(args.subgroup_values_to_keep)
    
    # NOTE: label_map cannot easily be passed via CLI, so we define a default here.
    # If you need to change the labels frequently, consider loading this dictionary
    # from an external JSON file within this script instead of hardcoding.
    label_map = {
        'A': 'Cohort A',
        'B': 'Cohort B',
        'C': 'Cohort C',
        'CellLine': 'Cell Line'
    }

    # Execute the plotting pipeline
    plot_umap_from_h5ad(
        h5ad_path=args.h5ad_path,
        output_dir=args.output_dir,
        batch_key=args.batch_key_user,
        cluster_key=args.cluster_key,
        label_map=label_map,
        group_key=args.group_key,
        group_values=group_values_to_keep_set,
        subgroup_key=args.subgroup_key,
        subgroup_values=subgroup_values_to_keep_set,
        background_label=args.background_color_label,
        subset_title=args.subset_title
    )


if __name__ == "__main__":
    main()
