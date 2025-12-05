import scanpy as sc
import pandas as pd
import numpy as np
import os
import argparse
import sys
from typing import List, Dict, Union, Any

# --- Configuration Constants ---
MIN_EXPRESSION_PCT = 0.10  # 10% minimum expression in at least one group
FDR_CUTOFF = 0.05          # False Discovery Rate (pvals_adj) threshold
LFC_CUTOFF = 0.25          # Minimum absolute Log Fold Change for significance
TARGET_GENE_COUNT = 0      # Target number of genes (1 up + 1 down) if using the count fallback
MIN_CELLS_PER_GROUP = 100  # Minimum required cells in BOTH comparison groups

# --- USER-DEFINED GROUPING ---
# These Response_Status categories will be combined into a single 'Aggressive' group.
AGGRESSIVE_STATUS_GROUPS = ['Persister', 'Recurrent', 'Residual', 'Resistant', 'Aggressive']
# These Response_Status categories will be combined into a single 'Control' (reference) group.
CONTROL_STATUS_GROUPS = ['less-Aggressive']  # Can add more if needed, e.g., ['less-Aggressive', 'non-Aggressive']
# The name of the new column created for DGE comparison
NEW_STATUS_KEY = 'aggressiveness_group'


def perform_dge_analysis(
    adata: sc.AnnData,
    cancer_type: str,
    cell_type_key: str,
    status_key: str,
    raw_layer: str,
    reference_group: str = 'Control',  # UPDATED: Changed to 'Control'
    output_dir: str = '.'
) -> Dict[str, Dict[str, Any]]:
    """
    Performs Wilcoxon Rank-Sum DGE and applies hybrid thresholding.
    """
    print(f"Starting DGE analysis using layer: '{raw_layer}'")
    print(f"Comparing groups in '{status_key}' with reference: '{reference_group}'")

    results_dfs = []
    top_genes_for_pathway: Dict[str, Dict[str, Any]] = {} 
    
    # REMOVED: Cancer type filtering moved to main()
    # adata = adata[adata.obs['Cancer_Type'] == 'BC'].copy()
    
    cell_types = adata.obs[cell_type_key].unique()

    for ct_name in cell_types:
        adata_subset = adata[adata.obs[cell_type_key] == ct_name].copy()
        
        groups_present = adata_subset.obs[status_key].unique()
        
        # Initial check for presence of both groups
        if not all(g in groups_present for g in ['Aggressive', reference_group]):
            print(f" -> Skipping {ct_name}: Missing one of the comparison groups ('Aggressive' or '{reference_group}').")
            continue
            
        # Check for minimum cell count in both groups
        group_counts = adata_subset.obs[status_key].value_counts()
        aggressive_count = group_counts.get('Aggressive', 0)
        control_count = group_counts.get(reference_group, 0)
        
        if aggressive_count < MIN_CELLS_PER_GROUP or \
           control_count < MIN_CELLS_PER_GROUP:
            print(f" -> Skipping {ct_name}: Group counts are too low for reliable DGE. Aggressive={aggressive_count}, Control={control_count}. (Min required: {MIN_CELLS_PER_GROUP})")
            continue

        print(f"Running Wilcoxon test for cell type: {ct_name}")

        # 1. Run scanpy.tl.rank_genes_groups on the subset
        sc.tl.rank_genes_groups(
            adata_subset,
            groupby=status_key,
            groups=['Aggressive'],
            reference=reference_group,
            method='wilcoxon',
            layer=raw_layer,
            key_added='wilcoxon_dge'
        )

        # 2. Extract and Filter (Expression)
        results_df = sc.get.rank_genes_groups_df(adata_subset, group='Aggressive', key='wilcoxon_dge')
        
        group_filter = adata_subset.obs[status_key] == 'Aggressive'
        ref_filter = adata_subset.obs[status_key] == reference_group
        
        raw_counts = adata_subset.layers[raw_layer]
        
        # Calculate pct_agg and pct_ctrl
        agg_subset = raw_counts[group_filter.values, :]
        ref_subset = raw_counts[ref_filter.values, :]
        results_df['pct_agg'] = np.mean(agg_subset.toarray() > 0, axis=0)
        results_df['pct_ctrl'] = np.mean(ref_subset.toarray() > 0, axis=0)
        
        results_df['cell_type'] = ct_name
        
        # 10% Expression Filter
        exp_filter = (results_df['pct_agg'] >= MIN_EXPRESSION_PCT) | \
                     (results_df['pct_ctrl'] >= MIN_EXPRESSION_PCT)
        
        filtered_df = results_df[exp_filter].copy()
        
        # 3. Hybrid Thresholding Logic
        
        # A. Get all statistically and biologically significant genes (FDR < 0.05 AND |LFC| > 1.00)
        statistically_significant_df = filtered_df[
            (filtered_df['pvals_adj'] < FDR_CUTOFF) &
            (abs(filtered_df['logfoldchanges']) > LFC_CUTOFF)
        ].copy()
        
        # Sort significant genes by LFC
        statistically_significant_df.sort_values(by='logfoldchanges', ascending=False, inplace=True)
        
        
        up_genes: List[str] = []
        down_genes: List[str] = []
        
        
        if len(statistically_significant_df) >= TARGET_GENE_COUNT:
            # Case 1: Enough highly significant genes found (show ALL significant)
            up_genes = list(statistically_significant_df[statistically_significant_df['logfoldchanges'] > 0]['names'])
            down_genes = list(statistically_significant_df[statistically_significant_df['logfoldchanges'] < 0]['names'])
            print(f" -> {ct_name}: Found {len(up_genes) + len(down_genes)} highly significant DEGs (FDR<{FDR_CUTOFF}, |LFC|>{LFC_CUTOFF}). Using ALL.")
        else:
            # Case 2: Use the Count Fallback (Top N up/down LFC from the expression-filtered list)
            # Sort the original expression-filtered list (all candidates)
            filtered_df.sort_values(by='logfoldchanges', ascending=False, inplace=True)
            
            N = TARGET_GENE_COUNT // 2
            
            # Top N up-regulated (positive LFC)
            up_regulated = filtered_df[filtered_df['logfoldchanges'] > 0].head(N)
            
            # Top N down-regulated (negative LFC)
            down_regulated = filtered_df[filtered_df['logfoldchanges'] < 0].tail(N)
            
            # Combine the lists for pathway analysis input
            up_genes = list(up_regulated['names'])
            down_genes = list(down_regulated['names'])
            
            print(f" -> {ct_name}: Found only {len(statistically_significant_df)} highly significant DEGs. Falling back to Top {len(up_genes)} Up and Top {len(down_genes)} Down by LFC.")

        # Combine the lists (Up first, then Down) and store the split point
        top_genes_list = up_genes + down_genes
        up_count = len(up_genes)

        top_genes_for_pathway[ct_name] = {'genes': top_genes_list, 'up_count': up_count}
        results_dfs.append(filtered_df)


    # 4. Save Final Outputs
    if not results_dfs:
        print("\nWarning: No cell types had both Aggressive and Control cells for comparison.")
        return {}
        
    full_results_path = os.path.join(output_dir, 'dge_wilcoxon_full_results.csv')
    pd.concat(results_dfs).to_csv(full_results_path, index=False)
    print(f"\n✅ Full DGE results saved to: {full_results_path}")

    top_genes_path = os.path.join(output_dir, 'dge_genes_for_pathway_analysis.csv')
    
    # Restructure for CSV: just the list of genes (up+down)
    gene_lists_only = {k: v['genes'] for k, v in top_genes_for_pathway.items()}
    max_len = max(len(v) for v in gene_lists_only.values()) if gene_lists_only else 0
    top_genes_df = pd.DataFrame(
        {k: pd.Series(v) for k, v in gene_lists_only.items()}
    ).fillna('')
    
    top_genes_df.to_csv(top_genes_path, index=False)
    print(f"✅ Genes for pathway analysis saved to: {top_genes_path}")

    return top_genes_for_pathway


def main():
    """Main function to parse arguments and execute the DGE pipeline."""
    parser = argparse.ArgumentParser(
        description="Perform Wilcoxon Rank-Sum DGE analysis on scRNA-seq data to find aggressiveness-associated genes.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- File/Path Arguments ---
    parser.add_argument(
        '--h5ad_path',
        type=str,
        required=True,
        help="Path to the input processed AnnData file (.h5ad)."
    )
    parser.add_argument(
        '--cancer_type',
        type=str,
        default=None,
        help="Cancer type to filter (e.g., 'BC'). If not specified, all cancer types will be included."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/f/H3K27/Data/GSEA/',
        help="Directory to save the DGE output files (CSV). Creates if it doesn't exist. (Default: dge_results)"
    )
    
    # --- Observation Key Arguments ---
    parser.add_argument(
        '--cell_type_key',
        type=str,
        default='cell_type',
        help="AnnData.obs column containing the cell type annotation (e.g., 'cell_type')."
    )
    parser.add_argument(
        '--status_key',
        type=str,
        default='Response_Status',
        help="AnnData.obs column containing the original aggressive status (e.g., 'Response_Status')."
    )
    parser.add_argument(
        '--raw_layer',
        type=str,
        default='raw_counts',
        help="AnnData layer containing the raw UMI counts (REQUIRED for DGE). (Default: raw_counts)"
    )


    args = parser.parse_args()

    # Load AnnData and create output directory
    try:
        adata = sc.read_h5ad(args.h5ad_path)
    except FileNotFoundError:
        print(f"Error: AnnData file not found at {args.h5ad_path}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes.")

    if args.raw_layer not in adata.layers:
        print(f"Error: Raw count layer '{args.raw_layer}' not found in adata.layers. DGE requires raw counts.")
        sys.exit(1)

    # ==========================================================================
    # FILTER BY CANCER TYPE (OPTIONAL - ONLY IF SPECIFIED)
    # ==========================================================================
    if args.cancer_type is not None:
        if 'Cancer_Type' in adata.obs.columns:
            print(f"\nFiltering data to Cancer_Type == '{args.cancer_type}'...")
            original_n_obs = adata.n_obs
            adata = adata[adata.obs['Cancer_Type'] == args.cancer_type].copy()
            print(f"Filtered from {original_n_obs} to {adata.n_obs} cells.")
            
            if adata.n_obs == 0:
                print(f"Error: No cells found with Cancer_Type == '{args.cancer_type}'")
                sys.exit(1)
        else:
            print(f"Warning: 'Cancer_Type' column not found in adata.obs. Skipping cancer type filtering.")
    else:
        print("\nNo --cancer_type specified. Including all cancer types in analysis.")

    # ==========================================================================
    # PREPROCESSING STEP: GROUPING MULTIPLE STATUSES INTO 'AGGRESSIVE' vs 'CONTROL'
    # ==========================================================================
    print(f"\nPreprocessing: Grouping statuses in '{args.status_key}'...")
    print(f"  - Aggressive group: {AGGRESSIVE_STATUS_GROUPS}")
    print(f"  - Control group: {CONTROL_STATUS_GROUPS}")

    adata.obs[NEW_STATUS_KEY] = 'Undefined'

    # Assign Control group
    control_mask = adata.obs[args.status_key].isin(CONTROL_STATUS_GROUPS)
    adata.obs.loc[control_mask, NEW_STATUS_KEY] = 'Control'

    # Assign Aggressive group
    agg_mask = adata.obs[args.status_key].isin(AGGRESSIVE_STATUS_GROUPS)
    adata.obs.loc[agg_mask, NEW_STATUS_KEY] = 'Aggressive'
    
    original_n_obs = adata.n_obs
    adata = adata[adata.obs[NEW_STATUS_KEY] != 'Undefined'].copy()
    if adata.n_obs < original_n_obs:
        print(f"Warning: {original_n_obs - adata.n_obs} cells were excluded because they did not belong to the defined Aggressive or Control groups.")
    
    if not all(group in adata.obs[NEW_STATUS_KEY].unique() for group in ['Aggressive', 'Control']):
        print("Error: After grouping, one or both comparison groups ('Aggressive', 'Control') are missing in the filtered data. Check input data and grouping configuration.")
        sys.exit(1)

    print(f"\nNew comparison groups created in column: '{NEW_STATUS_KEY}'")
    print(adata.obs[NEW_STATUS_KEY].value_counts())
    
    # Show breakdown of original statuses -> new groups
    print("\n--- Mapping of Original Status to New Groups ---")
    status_mapping = adata.obs.groupby([args.status_key, NEW_STATUS_KEY]).size().unstack(fill_value=0)
    print(status_mapping.to_markdown(numalign="left", stralign="left"))
    print("------------------------------------------------------------------")
    
    # ==========================================================================
    # PRINTING CELL COUNTS PER CELL TYPE
    # ==========================================================================
    print("\n--- Cell Counts per Cell Type (Aggressive vs. Control) ---")
    
    cell_type_counts = adata.obs.groupby([args.cell_type_key, NEW_STATUS_KEY]).size().unstack(fill_value=0)
    
    # Ensure both columns exist even if one is zero across the board
    if 'Aggressive' not in cell_type_counts.columns:
        cell_type_counts['Aggressive'] = 0
    if 'Control' not in cell_type_counts.columns:
        cell_type_counts['Control'] = 0
        
    # Calculate Total and format
    cell_type_counts['Total'] = cell_type_counts['Aggressive'] + cell_type_counts['Control']
    cell_type_counts = cell_type_counts[['Aggressive', 'Control', 'Total']]
    
    # Print the formatted table
    print(cell_type_counts.to_markdown(numalign="left", stralign="left"))
    print("------------------------------------------------------------------")
    # ==========================================================================

    # Execute the DGE analysis using the newly created, simplified column
    top_genes = perform_dge_analysis(
        adata=adata,
        cancer_type=args.cancer_type,
        cell_type_key=args.cell_type_key,
        status_key=NEW_STATUS_KEY,  # Use the new, simplified grouping column
        raw_layer=args.raw_layer,
        reference_group='Control',  # UPDATED: Changed to 'Control'
        output_dir=args.output_dir
    )

    print("\n\n--- Genes Selected for Pathway Analysis ---")
    for ct, data in top_genes.items():
        genes = data['genes']
        up_count = data['up_count']
        
        up_genes = genes[:up_count]
        down_genes = genes[up_count:]
        
        print(f"Cell Type: {ct} ({len(genes)} genes selected)")
        
        if up_genes:
            print(f"  Up-regulated ({len(up_genes)}): {', '.join(up_genes)}")
        if down_genes:
            print(f"  Down-regulated ({len(down_genes)}): {', '.join(down_genes)}")
        print("-" * 20)


if __name__ == "__main__":
    import sys
    if 'ipykernel' in sys.modules or 'jupyter' in sys.modules:
        print("Note: Running in an interactive environment. To run the full script, use: python dge_aggressiveness_analysis.py --h5ad_path <path/to/data.h5ad> ...")
    else:
        main()