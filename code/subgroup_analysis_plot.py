import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union, List, Optional

def filter_expressed_genes(adata, genes, min_cells=50, min_expr=0.1):
    """Keep only genes expressed in at least min_cells with expression > min_expr"""
    expressed = []
    for gene in genes:
        if gene not in adata.var_names:
            print(f"  ❌ Skipping {gene}: not found in dataset")
            continue
            
        # Handle sparse matrix
        gene_data = adata.X[:, adata.var_names == gene]
        if hasattr(gene_data, 'toarray'):
            gene_expr = gene_data.toarray().flatten()
        else:
            gene_expr = gene_data.flatten()
        
        n_expressing = np.sum(gene_expr > min_expr)
        if n_expressing >= min_cells:
            expressed.append(gene)
            print(f"  ✓ Keeping {gene}: {n_expressing} cells express it (mean={np.mean(gene_expr[gene_expr > 0]):.2f})")
        else:
            print(f"  ⚠️  Skipping {gene}: only {n_expressing} cells express it")
    
    return expressed

def plot_subgroup_analysis(
    input_h5ad: str,
    GROUP_KEY: Optional[str] = None,  # Changed to Optional
    GROUP_VALUES_TO_KEEP: Union[List[str], str] = 'all',
    SUBGROUP_KEY: Optional[str] = None,
    SUBGROUP_VALUES_TO_KEEP: Optional[Union[List[str], str]] = None,
    plot_type: str = 'stacked_bar',
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    dot_size_max: Optional[float] = None,
    enrichment_file_path: Optional[str] = None,
    output_filename: str = 'subgroup_analysis_plot.png'
):
    """
    Analyzes and plots single-cell data based on user-defined grouping and subgrouping keys.
    
    Args:
        input_h5ad: Path to the AnnData h5ad file
        GROUP_KEY: Column name for primary grouping (optional for enrichment_bar, dot_gene)
        GROUP_VALUES_TO_KEEP: Values to keep from GROUP_KEY ('all' or list)
        SUBGROUP_KEY: Column name for subgrouping/filtering
        SUBGROUP_VALUES_TO_KEEP: Values to keep from SUBGROUP_KEY
        plot_type: 'stacked_bar', 'violin_gene', 'dot_gene', 'umap_group', 'enrichment_bar'
        genes: List of genes (required for violin_gene, dot_gene)
        layer: Optional layer for expression values
        dot_size_max: Max fraction (0-1) for dot size legend
        enrichment_file_path: Path to enrichment CSV/TSV file
        output_filename: Output filename
    """
    
    # Validate parameters based on plot type
    if plot_type in ['violin_gene', 'dot_gene'] and not genes:
        print(f"❌ Error: 'genes' parameter is required for {plot_type}.")
        return
    
    if plot_type == 'enrichment_bar' and not enrichment_file_path:
        print("❌ Error: 'enrichment_file_path' is required for enrichment_bar plot.")
        return
    
    # GROUP_KEY is required for certain plot types
    if plot_type in ['stacked_bar', 'violin_gene', 'umap_group'] and not GROUP_KEY:
        print(f"❌ Error: 'GROUP_KEY' is required for {plot_type}.")
        return
    
    # 1. Data Loading
    try:
        adata = sc.read_h5ad(input_h5ad)
    except FileNotFoundError:
        print(f"❌ Error: AnnData file not found at {input_h5ad}")
        return
    
    # 2. Apply GROUP filtering (if GROUP_KEY provided)
    if GROUP_KEY and GROUP_KEY in adata.obs:
        if isinstance(GROUP_VALUES_TO_KEEP, list):
            adata = adata[adata.obs[GROUP_KEY].isin(GROUP_VALUES_TO_KEEP)].copy()
        elif isinstance(GROUP_VALUES_TO_KEEP, str) and GROUP_VALUES_TO_KEEP.lower() != 'all':
            adata = adata[adata.obs[GROUP_KEY] == GROUP_VALUES_TO_KEEP].copy()

    # 3. Apply SUBGROUP filtering (if SUBGROUP_KEY provided)
    if SUBGROUP_KEY and SUBGROUP_KEY in adata.obs and SUBGROUP_VALUES_TO_KEEP is not None:
        if isinstance(SUBGROUP_VALUES_TO_KEEP, list):
            adata = adata[adata.obs[SUBGROUP_KEY].isin(SUBGROUP_VALUES_TO_KEEP)].copy()
        elif isinstance(SUBGROUP_VALUES_TO_KEEP, str) and SUBGROUP_VALUES_TO_KEEP.lower() != 'all':
            adata = adata[adata.obs[SUBGROUP_KEY] == SUBGROUP_VALUES_TO_KEEP].copy()

    if adata.n_obs == 0:
        print("❌ Error: Filtering resulted in zero cells. Check your filter values.")
        return

    # 4. Filter genes if provided
    if genes:
        missing_genes = [g for g in genes if g not in adata.var_names]
        if missing_genes:
            print(f"⚠️  Warning: Genes not found and skipped: {missing_genes}")
        genes = [g for g in genes if g in adata.var_names]
        if not genes:
            print("❌ Error: No valid genes remaining after filtering.")
            return

    # 5. Generate plots
    
    if plot_type == 'stacked_bar':
        print(f"📊 Generating stacked bar plot for {GROUP_KEY} colored by {SUBGROUP_KEY}...")
        
        counts_table = pd.crosstab(
            adata.obs[GROUP_KEY], 
            adata.obs[SUBGROUP_KEY],
            normalize='index' 
        ) * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        counts_table.plot(
            kind='bar', stacked=True, ax=ax,
            colormap='viridis', edgecolor='black', linewidth=0.5
        )

        ax.set_title(f'Proportion of {GROUP_KEY} Subgroups by {SUBGROUP_KEY}')
        ax.set_xlabel(GROUP_KEY)
        ax.set_ylabel(f'Proportion of Cells in {GROUP_KEY} (%)')
        ax.legend(title=SUBGROUP_KEY, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved: '{output_filename}'")
        plt.close(fig)

    elif plot_type == 'enrichment_bar':
        print(f"📊 Generating enrichment bar plot...")
        
        # Load enrichment data
        try:
            if enrichment_file_path.endswith('.csv'):
                enrichment_df = pd.read_csv(enrichment_file_path)
            elif enrichment_file_path.endswith(('.tsv', '.txt')):
                enrichment_df = pd.read_csv(enrichment_file_path, sep='\t')
            else:
                print("❌ Error: Enrichment file must be .csv, .tsv, or .txt.")
                return
        except FileNotFoundError:
            print(f"❌ Error: Enrichment file not found at {enrichment_file_path}")
            return
        
        # Validate columns
        required_cols = ['Term'] # Only Term is strictly required, Category will be handled.
        category_col = 'Gene_set' # We will use this column for categorization
        
    # 1. Validation and Column Renaming
        qval_cols = [col for col in enrichment_df.columns  
                     if any(x in col.lower() for x in ['q-value', 'fdr', 'p-adj', 'qval', 'adjusted p'])]
        
        # Check for 'Term' and 'Gene_set' (our desired category)
        if 'Term' not in enrichment_df.columns or 'Gene_set' not in enrichment_df.columns:
            print(f"❌ Error: Enrichment file must contain 'Term' and 'Gene_set' columns.")
            print(f"    Found columns: {enrichment_df.columns.tolist()}")
            return
        
        # Check for the significance column
        if not qval_cols:
            print(f"❌ Error: No Q-value/FDR/P-adjusted column found.")
            print(f"    Found columns: {enrichment_df.columns.tolist()}")
            return
            
        # --- Renaming 'Gene_set' to 'Category' for plotting compatibility ---
        enrichment_df = enrichment_df.rename(columns={'Gene_set': 'Category'}) 
    
        qval_col = qval_cols[0]
        print(f"    Using Q-value column: '{qval_col}'")
        
        # 2. Calculate -log10(q-value) (using the found qval_col)
        enrichment_df['Neg_Log_Qval'] = -np.log10(enrichment_df[qval_col].astype(float))
        enrichment_df = enrichment_df[np.isfinite(enrichment_df['Neg_Log_Qval'])]
        enrichment_df = enrichment_df[enrichment_df['Neg_Log_Qval'] > 0]
        enrichment_df = enrichment_df.sort_values(by='Neg_Log_Qval', ascending=False).head(15)
        
        if enrichment_df.empty:
            print("⚠️  Warning: No significant enrichment terms found.")
            return
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        categories = enrichment_df['Category'].unique()
        color_map = plt.cm.get_cmap('Set1', len(categories))
        category_colors = {cat: color_map(i) for i, cat in enumerate(categories)}
        
        ax.barh(
            enrichment_df['Term'], 
            enrichment_df['Neg_Log_Qval'], 
            color=[category_colors[c] for c in enrichment_df['Category']],
            edgecolor='black', linewidth=0.5
        )
        
        subset_name = (SUBGROUP_VALUES_TO_KEEP 
                      if isinstance(SUBGROUP_VALUES_TO_KEEP, str) and SUBGROUP_VALUES_TO_KEEP.lower() != 'all'
                      else 'Overall Cohort')
        ax.set_title(f'Top Enriched Pathways in {subset_name}', fontsize=14, pad=15)
        ax.set_xlabel('$-\log_{10}$ q-value', fontsize=12)
        ax.set_ylabel('Enriched Term', fontsize=12)
        
        sig_threshold = 2.0
        ax.axvline(x=sig_threshold, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(sig_threshold + 0.2, len(enrichment_df) - 0.5, 
                f'$-\log_{{10}}(q) = {sig_threshold}$', color='gray', ha='left', va='top')
        
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        
        legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=category_colors[cat], edgecolor='black') 
                         for cat in categories]
        ax.legend(legend_handles, categories, title="Category",
                 bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved: '{output_filename}'")
        plt.close(fig)

    elif plot_type == 'umap_group':
        print(f"🗺️  Generating UMAP plot colored by {GROUP_KEY}...")
        
        if 'X_umap' not in adata.obsm.keys():
            print("❌ Error: UMAP coordinates not found. Run sc.tl.umap(adata) first.")
            return
        
        adata.obs[GROUP_KEY] = adata.obs[GROUP_KEY].astype('category')
        
        fig, ax = plt.subplots(figsize=(8, 8))
        sc.pl.umap(adata, color=GROUP_KEY, palette='tab20',
                  legend_loc='on data', title=f"UMAP colored by {GROUP_KEY}",
                  ax=ax, show=False, frameon=False)
        
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved: '{output_filename}'")
        plt.close(fig)
        
    elif plot_type == 'violin_gene':
        subset_name = (SUBGROUP_VALUES_TO_KEEP 
                      if isinstance(SUBGROUP_VALUES_TO_KEEP, str) and SUBGROUP_VALUES_TO_KEEP.lower() != 'all'
                      else 'All Samples')
        print(f"🎻 Generating violin plots for {len(genes)} genes in {subset_name} grouped by {GROUP_KEY}...")
        
        n_genes = len(genes)
        fig, axes = plt.subplots(1, n_genes, figsize=(4.5 * n_genes, 8)) 
        axes = np.ravel(axes) if n_genes > 1 else [axes]

        for i, gene in enumerate(genes):
            sc.pl.violin(adata, keys=gene, groupby=GROUP_KEY,
                        ax=axes[i], layer=layer, show=False,
                        rotation=90, stripplot=False, size=2)
        
        plt.suptitle(f'Expression Distribution in {subset_name} by {GROUP_KEY}', y=0.98, fontsize=16)
        plt.subplots_adjust(top=0.9, bottom=0.25)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved: '{output_filename}'")
        plt.close(fig)
        
    elif plot_type == 'dot_gene':
        subset_name = (SUBGROUP_VALUES_TO_KEEP 
                      if isinstance(SUBGROUP_VALUES_TO_KEEP, str) and SUBGROUP_VALUES_TO_KEEP.lower() != 'all'
                      else 'All Samples')
        
        # Use SUBGROUP_KEY as the grouping variable for dot plot if no GROUP_KEY
        groupby_key = GROUP_KEY if GROUP_KEY else SUBGROUP_KEY
        
        if not groupby_key or groupby_key not in adata.obs:
            print("❌ Error: Need either GROUP_KEY or SUBGROUP_KEY for dot plot grouping.")
            return
        
        print(f"🔵 Generating dot plot for {len(genes)} genes grouped by {groupby_key}...")
        
        n_genes = len(genes)
        n_groups = len(adata.obs[groupby_key].unique())
        width = max(4, 1.2 * n_genes)
        height = max(3, 0.7 * n_groups + 3)
        
        max_fraction_to_use = dot_size_max
        if max_fraction_to_use is not None and not (0 <= max_fraction_to_use <= 1):
            print("⚠️  Warning: dot_size_max must be between 0 and 1. Using auto-scaling.")
            max_fraction_to_use = None
        
        dp = sc.pl.dotplot(
            adata, var_names=genes, groupby=groupby_key,
            layer=layer, color_map='RdBu_r', standard_scale='var',
            figsize=(width, height), dot_max=max_fraction_to_use,
            show=False, save=False
        )
        
        fig = dp.fig if hasattr(dp, 'fig') else plt.gcf()
        fig.suptitle(f"Gene Expression in {subset_name} by {groupby_key}", y=1.02, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(left=0.25, right=0.85, top=0.9, bottom=0.2)
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved: '{output_filename}'")
        plt.close(fig)

    else:
        print(f"❌ Error: Unknown plot_type '{plot_type}'.")
        print("   Choose from: 'stacked_bar', 'violin_gene', 'dot_gene', 'umap_group', 'enrichment_bar'")
        return


if __name__ == '__main__':
    H5AD_FILE_PATH = '/mnt/f/H3K27/Data/scRNA/scRNA_final.h5ad'
    
    # Load data once to filter genes
    print("🔍 Loading data to filter expressed genes...\n")
    adata_temp = sc.read_h5ad(H5AD_FILE_PATH)
    
    CANCER_LOGIC = [
        # Immune Checkpoints
        'PDCD1', 'CD274', 'CTLA4', 'CD58',
        # Efflux/Detoxification
        'ABCB1', 'ABCG2', 'ABCC1',
        # EMT
        'CDH1', 'VIM', 'ZEB1', 'FN1',
        # Metabolic
        'GLS', 'NDUFS1', 'HK2',
        # Senescence & SASP
        'CDKN1A', 'CDKN2A', 'IL6', 'IL8', 'TNF',
        # Survival Pathways
        'AKT1', 'FOXO3', 'STAT3', 'NFKB1',
        # Cell Cycle
        'AURKA', 'DLGAP5', 'BUB1B', 'KIF20A', 'CENPF', 'CENPA', 'CCND1', 'CCNE1',
        # DNA Damage Repair
        'ATM', 'ATR', 'BRCA1', 'BRCA2', 'BCL2', 'BCL2L1', 'BAX',
        # Mechanotransduction
        'YAP1', 'WWTR1', 'ITGB1', 'ITGA6', 'PTK2', 'COL1A1', 'LOX', 'ACTN4',
    ]
    
    # Filter genes once globally
    CANCER_LOGIC_FILTERED = filter_expressed_genes(adata_temp, CANCER_LOGIC, min_cells=100, min_expr=0.1)
    del adata_temp  # Free memory
    
    print(f"\n✅ {len(CANCER_LOGIC_FILTERED)}/{len(CANCER_LOGIC)} genes passed filtering\n")
    print("="*80 + "\n")
    
    # Iterate through cancer types
    for CANCER in ['BC', 'OC', 'EC', 'GC', 'Normal Breast Cells']:
        print(f"🔬 Processing {CANCER}...")
        print("-" * 80)
        
        # # Violin plot (grouped by Cancer_Status)
        # plot_subgroup_analysis(
        #     input_h5ad=H5AD_FILE_PATH,
        #     GROUP_KEY='Cancer_Status',
        #     GROUP_VALUES_TO_KEEP='all',
        #     SUBGROUP_KEY='Cancer_Type',
        #     SUBGROUP_VALUES_TO_KEEP=CANCER,
        #     plot_type='violin_gene',
        #     genes=CANCER_LOGIC_FILTERED,
        #     output_filename=f'violin_{CANCER}.png'
        # )
        
        # Dot plot (grouped by Cancer_Type itself, showing only this cancer)
        plot_subgroup_analysis(
            input_h5ad=H5AD_FILE_PATH,
            GROUP_KEY='Cancer_Status',  # No GROUP_KEY needed
            SUBGROUP_KEY='Cancer_Type',
            SUBGROUP_VALUES_TO_KEEP=CANCER,
            plot_type='dot_gene',
            genes=CANCER_LOGIC_FILTERED,
            output_filename=f'dot_{CANCER}.png'
        )
        
        # # Enrichment bar plot (no grouping needed)
        # plot_subgroup_analysis(
        #     input_h5ad=H5AD_FILE_PATH,
        #     GROUP_KEY=None,  # Not needed for enrichment
        #     SUBGROUP_KEY='Cancer_Type',
        #     SUBGROUP_VALUES_TO_KEEP=CANCER,
        #     plot_type='enrichment_bar',
        #     enrichment_file_path=f'/mnt/f/H3K27/GSEA/c2.{CANCER}.txt',
        #     output_filename=f'enrichment_bar_{CANCER}.png'
        # )
        
        print("\n" + "="*80 + "\n")