import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union, List, Optional
import gseapy as gp
from scipy import sparse

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


def perform_deg_and_enrichment(
    input_h5ad: str,
    subgroup_key: str,
    test_group: str,
    control_group: str = 'Non-malignant',
    cancer_type_key: str = None,
    cancer_type_value: str = None,
    gmt_file: str = '/mnt/f/H3K27/GSEA/c2.all.v2023.1.Hs.symbols.gmt',
    output_dir: str = './',
    n_top_genes: int = 500,
    logfc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
    method: str = 'wilcoxon'
):
    """
    Performs differential expression analysis and gene set enrichment analysis.
    
    Args:
        input_h5ad: Path to the AnnData h5ad file
        subgroup_key: Column name for comparison (e.g., 'Cancer_Status')
        test_group: Test group value (e.g., 'Malignant')
        control_group: Control group value (e.g., 'Non-malignant')
        cancer_type_key: Optional column to filter by cancer type (e.g., 'Cancer_Type')
        cancer_type_value: Value to filter (e.g., 'BC')
        gmt_file: Path to GMT file for enrichment
        output_dir: Output directory for results
        n_top_genes: Number of top genes to use for ORA
        logfc_threshold: Log fold change threshold for filtering DEGs
        pval_threshold: P-value threshold for filtering DEGs
        method: Statistical test method ('wilcoxon', 't-test', 't-test_overestim_var')
    
    Returns:
        dict: Dictionary containing DEG results and enrichment results
    """
    
    print(f"\n{'='*80}")
    print(f"🔬 Analyzing: {test_group} vs {control_group}")
    if cancer_type_value:
        print(f"   Cancer Type: {cancer_type_value}")
    print(f"{'='*80}\n")
    
    # 1. Load data
    print("📂 Loading data...")
    try:
        adata = sc.read_h5ad(input_h5ad)
    except FileNotFoundError:
        print(f"❌ Error: AnnData file not found at {input_h5ad}")
        return None
    
    # 2. Filter by cancer type first if specified
    if cancer_type_key and cancer_type_value:
        print(f"🔍 Filtering to {cancer_type_value} samples...")
        if cancer_type_key not in adata.obs.columns:
            print(f"❌ Error: '{cancer_type_key}' not found in adata.obs")
            return None
        adata = adata[adata.obs[cancer_type_key] == cancer_type_value].copy()
        print(f"   Cells from {cancer_type_value}: {adata.n_obs}")
    
    # 3. Filter to relevant comparison groups
    print(f"🔍 Filtering to {test_group} and {control_group}...")
    if subgroup_key not in adata.obs.columns:
        print(f"❌ Error: '{subgroup_key}' not found in adata.obs")
        return None
    
    adata_subset = adata[adata.obs[subgroup_key].isin([test_group, control_group])].copy()
    
    if adata_subset.n_obs == 0:
        print(f"❌ Error: No cells found for groups '{test_group}' or '{control_group}'")
        return None
    
    print(f"   {test_group}: {sum(adata_subset.obs[subgroup_key] == test_group)} cells")
    print(f"   {control_group}: {sum(adata_subset.obs[subgroup_key] == control_group)} cells")
    
    # 3. Perform differential expression analysis
    print(f"\n📊 Performing differential expression analysis ({method})...")
    
    # Rank genes for each group vs rest
    sc.tl.rank_genes_groups(
        adata_subset,
        groupby=subgroup_key,
        groups=[test_group],
        reference=control_group,
        method=method,
        use_raw=False
    )
    
    # Extract results
    deg_results = sc.get.rank_genes_groups_df(adata_subset, group=test_group)
    
    # Save all DEG results
    suffix = f"_{cancer_type_value}" if cancer_type_value else ""
    deg_output = f"{output_dir}/DEG_{test_group}_vs_{control_group}{suffix}.csv"
    deg_results.to_csv(deg_output, index=False)
    print(f"   ✅ Saved DEG results: {deg_output}")
    
    # 4. Filter significant genes
    print(f"\n🔬 Filtering DEGs (logFC > {logfc_threshold}, p-adj < {pval_threshold})...")
    
    # Upregulated genes
    up_genes = deg_results[
        (deg_results['logfoldchanges'] > logfc_threshold) & 
        (deg_results['pvals_adj'] < pval_threshold)
    ].sort_values('logfoldchanges', ascending=False)
    
    # Downregulated genes
    down_genes = deg_results[
        (deg_results['logfoldchanges'] < -logfc_threshold) & 
        (deg_results['pvals_adj'] < pval_threshold)
    ].sort_values('logfoldchanges', ascending=True)
    
    print(f"   ✓ Upregulated genes: {len(up_genes)}")
    print(f"   ✓ Downregulated genes: {len(down_genes)}")
    
    if len(up_genes) == 0 and len(down_genes) == 0:
        print("   ⚠️  Warning: No significant DEGs found. Adjusting thresholds...")
        # Relax thresholds
        up_genes = deg_results[deg_results['logfoldchanges'] > 0].sort_values(
            'logfoldchanges', ascending=False
        ).head(n_top_genes)
        down_genes = deg_results[deg_results['logfoldchanges'] < 0].sort_values(
            'logfoldchanges', ascending=True
        ).head(n_top_genes)
        print(f"   ✓ Using top {len(up_genes)} up and {len(down_genes)} down genes")
    
    # 5. Run Gene Set Enrichment Analysis (GSEA)
    print(f"\n🧬 Running GSEA...")
    
    # Prepare ranked gene list for GSEA
    gene_rank = deg_results[['names', 'logfoldchanges']].copy()
    gene_rank.columns = ['gene', 'rank']
    gene_rank = gene_rank.sort_values('rank', ascending=False)
    
    try:
        gsea_results = gp.prerank(
            rnk=gene_rank,
            gene_sets=gmt_file,
            threads=4,
            permutation_num=1000,
            outdir=None,
            seed=42,
            verbose=False
        )
        
        # Save GSEA results
        gsea_output = f"{output_dir}/GSEA_{test_group}_vs_{control_group}{suffix}.csv"
        gsea_results.res2d.to_csv(gsea_output, index=False)
        print(f"   ✅ Saved GSEA results: {gsea_output}")
        
    except Exception as e:
        print(f"   ⚠️  GSEA failed: {str(e)}")
        gsea_results = None
    
    # 6. Run Over-Representation Analysis (ORA) on top upregulated genes
    print(f"\n📈 Running ORA on top upregulated genes...")
    
    top_up_genes = up_genes.head(n_top_genes)['names'].tolist()
    
    if len(top_up_genes) > 0:
        try:
            ora_results = gp.enrichr(
                gene_list=top_up_genes,
                gene_sets=gmt_file,
                outdir=None,
                verbose=False
            )
            
            # Save ORA results
            ora_output = f"{output_dir}/ORA_{test_group}_vs_{control_group}{suffix}.csv"
            ora_results.res2d.to_csv(ora_output, index=False)
            print(f"   ✅ Saved ORA results: {ora_output}")
            
        except Exception as e:
            print(f"   ⚠️  ORA failed: {str(e)}")
            ora_results = None
    else:
        print("   ⚠️  No upregulated genes for ORA")
        ora_results = None
    
    # 7. Create visualization
    print(f"\n🎨 Creating enrichment plot...")
    plot_enrichment_results(
        gsea_results=gsea_results,
        ora_results=ora_results,
        test_group=test_group,
        control_group=control_group,
        cancer_type=cancer_type_value,
        output_filename=f"{output_dir}/enrichment_{test_group}{suffix}.png"
    )
    
    return {
        'deg': deg_results,
        'up_genes': up_genes,
        'down_genes': down_genes,
        'gsea': gsea_results,
        'ora': ora_results
    }


def plot_enrichment_results(
    gsea_results=None,
    ora_results=None,
    test_group='',
    control_group='',
    cancer_type=None,
    output_filename='enrichment_plot.png',
    top_n=15
):
    """Plot enrichment results from GSEA or ORA"""
    
    # Determine which results to plot
    if gsea_results is not None and gsea_results.res2d is not None:
        df = gsea_results.res2d.copy()
        df = df.sort_values('NES', ascending=False).head(top_n)
        value_col = 'NES'
        title_suffix = f' ({cancer_type})' if cancer_type else ''
        title = f'GSEA: {test_group} vs {control_group}{title_suffix}'
        xlabel = 'Normalized Enrichment Score (NES)'
    elif ora_results is not None and ora_results.res2d is not None:
        df = ora_results.res2d.copy()
        df = df.sort_values('Adjusted P-value', ascending=True).head(top_n)
        df['Neg_Log_Qval'] = -np.log10(df['Adjusted P-value'])
        value_col = 'Neg_Log_Qval'
        title_suffix = f' ({cancer_type})' if cancer_type else ''
        title = f'ORA: {test_group} vs {control_group}{title_suffix}'
        xlabel = '$-\\log_{10}$ (Adjusted P-value)'
    else:
        print("   ⚠️  No enrichment results to plot")
        return
    
    if df.empty:
        print("   ⚠️  No significant enrichment terms found")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Clean up term names (remove prefix like "KEGG_")
    df['Term_Clean'] = df['Term'].str.replace(r'^[A-Z]+_', '', regex=True)
    df['Term_Clean'] = df['Term_Clean'].str.replace('_', ' ')
    
    # Plot
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(df)))
    
    ax.barh(
        df['Term_Clean'],
        df[value_col],
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Pathway', fontsize=12)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    ax.invert_yaxis()
    
    # Add significance threshold line for ORA
    if value_col == 'Neg_Log_Qval':
        sig_threshold = 1.301  # p=0.05
        ax.axvline(x=sig_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(sig_threshold + 0.1, len(df) - 0.5, 
                'p = 0.05', color='red', ha='left', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Plot saved: {output_filename}")
    plt.close(fig)


if __name__ == '__main__':
    H5AD_FILE_PATH = '/mnt/f/H3K27/Data/scRNA/scRNA_final.h5ad'
    GMT_FILE = '/mnt/f/H3K27/GSEA/c2.all.v2023.1.Hs.symbols.gmt'
    OUTPUT_DIR = '/mnt/f/H3K27/GSEA/results'
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # First, let's check what cell types we have
    print("🔍 Checking available cell types...")
    adata_check = sc.read_h5ad(H5AD_FILE_PATH)
    print("\n--- Unique cell_type values ---")
    print(adata_check.obs['cell_type'].unique())
    print("\n--- Cell type counts ---")
    print(adata_check.obs['cell_type'].value_counts())
    print("\n" + "="*80 + "\n")
    
    # OPTION A: Compare cancer cells vs non-cancer cells
    # Cancer cells are labeled by cancer type: BC, OC, EC, GC
    CANCER_CELL_TYPES = ['BC', 'OC', 'EC', 'GC']
    
    # Non-cancer cells (stromal, immune, normal epithelial)
    NON_CANCER_CELL_TYPES = [
        'Fibroblasts', 'Myofibroblasts', 'Endothelial', 'Smooth_Muscle', 'Pericytes', 'Mesothelial',  # Stromal
        'CD8_T', 'CD4_T', 'Tregs', 'NK_Cells', 'B_Cells', 'Plasma',  # Immune
        'Macrophages', 'Monocytes', 'Dendritic', 'Mast',  # Myeloid
        'Normal_Epithelial', 'Adipocytes'  # Other
    ]
    
    # Define cancer types to analyze
    CANCER_TYPES = ['Breast', 'Ovarian', 'Endometrial', 'Gastric']
    
    # Dictionary to store all results
    all_results = {}
    
    print("\n" + "#"*80)
    print("# APPROACH: Comparing Cancer/Epithelial vs Stromal/Immune cells")
    print("# within each cancer type")
    print("#"*80 + "\n")
    
    # Create a temporary column that groups cell types
    print("📝 Creating cancer vs non-cancer grouping...")
    adata_check.obs['cell_group'] = 'Other'
    
    # Mark cancer cells (exact match)
    adata_check.obs.loc[adata_check.obs['cell_type'].isin(CANCER_CELL_TYPES), 'cell_group'] = 'Cancer'
    
    # Mark non-cancer cells (exact match)
    adata_check.obs.loc[adata_check.obs['cell_type'].isin(NON_CANCER_CELL_TYPES), 'cell_group'] = 'Non-Cancer'
    
    print("\nCell group distribution:")
    print(adata_check.obs['cell_group'].value_counts())
    
    # Check distribution per cancer type
    print("\nDistribution by Cancer Type:")
    for cancer in CANCER_TYPES:
        cancer_subset = adata_check[adata_check.obs['Cancer_Type'] == cancer]
        print(f"\n{cancer}:")
        print(cancer_subset.obs['cell_group'].value_counts())
    
    # Save this grouping back to the h5ad
    print("\n💾 Saving updated h5ad with cell_group column...")
    adata_check.write_h5ad(H5AD_FILE_PATH)
    
    print("\n" + "="*80 + "\n")
    
    # Now run the analysis
    # Compare each cancer cell type vs non-cancer cells from the same cancer type samples
    CANCER_TYPES_TO_ANALYZE = ['BC', 'OC', 'EC', 'GC']
    
    for cancer_type in CANCER_TYPES_TO_ANALYZE:
        print(f"\n{'#'*80}")
        print(f"# Processing: {cancer_type}")
        print(f"# Comparing: Cancer cells vs Non-Cancer cells within {cancer_type}")
        print(f"{'#'*80}")
        
        # Check if this cancer type exists in the data
        cancer_samples = adata_check[adata_check.obs['Cancer_Type'] == cancer_type]
        if len(cancer_samples) == 0:
            print(f"⚠️  Skipping {cancer_type}: no samples found in dataset")
            continue
        
        results = perform_deg_and_enrichment(
            input_h5ad=H5AD_FILE_PATH,
            subgroup_key='cell_group',         # Use our new grouping
            test_group='Cancer',               # Cancer cells
            control_group='Non-Cancer',        # Stromal/immune cells
            cancer_type_key='Cancer_Type',     # Filter by cancer type
            cancer_type_value=cancer_type,     # BC, OC, EC, or GC
            gmt_file=GMT_FILE,
            output_dir=OUTPUT_DIR,
            n_top_genes=500,
            logfc_threshold=0.5,
            pval_threshold=0.05,
            method='wilcoxon'
        )
        
        all_results[cancer_type] = results
        print(f"\n✅ Completed analysis for {cancer_type}\n")
    
    # Clean up
    del adata_check
    
    print(f"\n{'='*80}")
    print("🎉 All analyses completed!")
    print(f"{'='*80}\n")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("\nFiles generated for each cancer type:")
    print("  - DEG_Cancer_vs_Non-Cancer_{cancer}.csv")
    print("  - GSEA_Cancer_vs_Non-Cancer_{cancer}.csv")
    print("  - ORA_Cancer_vs_Non-Cancer_{cancer}.csv")
    print("  - enrichment_Cancer_{cancer}.png")