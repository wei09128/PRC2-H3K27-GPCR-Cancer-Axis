import pandas as pd
import numpy as np
import gseapy as gp
from gseapy.parser import read_gmt   # <-- add this

# Load your pseudobulk DGE data
df = pd.read_csv("/mnt/f/H3K27/Data/CCC_pseudo_BC/summary/pseudobulk_DGE_master.csv")
# df = pd.read_csv("/mnt/f/H3K27/Data/CCC/dge_wilcoxon_full_results.csv")


# Filter for BC patient-level comparisons (adjust the filter based on your column names)
# You'll need to adjust this based on how your data marks patient-level BC comparisons
bc_patient = df[
    (df['cell_type'].notna())# &  # has cell type info
    # Add your specific filter for BC patient-level aggressive vs non-aggressive
    # For example: (df['comparison'] == 'BC_aggressive_vs_nonaggressive')
].copy()

print(f"Total BC patient-level DEGs: {len(bc_patient)}")
print(f"Significant DEGs (FDR < 0.05): {bc_patient['FDR'].lt(0.05).sum()}")
print(f"\nCell types present:\n{bc_patient['cell_type'].value_counts()}")


gmt_path = "/mnt/f/H3K27/Data/CCC/c2.all.v2023.1.Hs.symbols.gmt"
gmt_dict = read_gmt(gmt_path)       # <-- change this line
kondo_full      = gmt_dict["KONDO_PROSTATE_CANCER_WITH_H3K27ME3"]
mikkelsen_mef   = gmt_dict["MIKKELSEN_MEF_HCP_WITH_H3K27ME3"]
mikkelsen_mcv6  = gmt_dict["MIKKELSEN_MCV6_HCP_WITH_H3K27ME3"]
benporath_es    = gmt_dict["BENPORATH_ES_WITH_H3K27ME3"]
nuytten_ezh2    = gmt_dict["NUYTTEN_EZH2_TARGETS_UP"]
nuytten_nipp1   = gmt_dict["NUYTTEN_NIPP1_TARGETS_UP"]
# Define PRC2 gene sets


PRC2_GENESETS = {
    # core subunits
    "PRC2_core": [
        "EZH2", "EED", "SUZ12", "RBBP4", "RBBP7"
    ],
    # Pan-gynecologic "core cancer" signature
    "PanGyn_top65_core": [
        "IL7R","FBLN2","LAMP3","ERAP2","GPR132","CD86","BGN","MRPL12","IL2RA",
        "TNFSF13B","IL18","BCL2A1","ITGB2","PXDN","DUSP8","AIF1","MYO1G","ATF3",
        "TUBA1B","ARIH2OS","CCR7","PCDH9","CLEC4E","CARHSP1","SRM","EREG","AQP9",
        "SDF2L1","GAL","LYZ","BHLHE41","CD24","SERPINF1","NR1D2","LRRC8A","SEC61G",
        "CST7","LXN","CHPT1","SYTL3","ATP2B1-AS1","CXXC5","CCL5","IRF8","LPAR6",
        "CCL22","FOS","MXRA8","MRPS15","LTC4S","KLF4","HLA-DQA1","NEURL3","RIN2",
        "RETREG1","GADD45A","TNRC6C","PLBD1","PDE7B","RHOF","PCAT19","MYDGF",
        "ALOX5AP","RGS2","EGR1"
    ],
    # 1) Immune remodeling module
    "PanGyn_immune_remodeling": [
        "IL7R","LAMP3","ERAP2","GPR132","CD86","IL2RA","TNFSF13B","IL18","BCL2A1",
        "AIF1","MYO1G","CCR7","CLEC4E","CCL5","IRF8","LPAR6","CCL22","HLA-DQA1",
        "CD24","LYZ","CST7","ITGB2","NEURL3"
    ],
    # 2) Matrisome / mechanical module
    "PanGyn_matrisome_mechanical": [
        "FBLN2","BGN","PXDN","MXRA8","SERPINF1","PCDH9"
    ],
    # 3) Metabolic / mitochondrial / lipid module
    "PanGyn_metabolic_reprogramming": [
        "AQP9","GAL","SRM","CHPT1","PLBD1","PDE7B","ALOX5AP","LTC4S",
        "MRPL12","MRPS15","MYDGF","LXN","LRRC8A"
    ],
    # 4) TF / stress / early-response module
    "PanGyn_TF_stress_response": [
        "ATF3","BHLHE41","NR1D2","CXXC5","FOS","KLF4","RGS2","EGR1","DUSP8",
        "GADD45A","TNRC6C","RHOF","RIN2","RETREG1","SYTL3","SEC61G","EREG",
        "SDF2L1","CARHSP1","TUBA1B","ARIH2OS","ATP2B1-AS1","PCAT19"
    ],
}

# Add other gene sets if you have them
if kondo_full:
    PRC2_GENESETS["KONDO_H3K27ME3"] = kondo_full
if mikkelsen_mef:
    PRC2_GENESETS["MIKKELSEN_MEF_H3K27ME3"] = mikkelsen_mef
if mikkelsen_mcv6:
    PRC2_GENESETS["MIKKELSEN_MCV6_H3K27ME3"] = mikkelsen_mcv6
if benporath_es:
    PRC2_GENESETS["BENPORATH_ES_H3K27ME3"] = benporath_es
if nuytten_ezh2:
    PRC2_GENESETS["NUYTTEN_EZH2_TARGETS_UP"] = nuytten_ezh2
if nuytten_nipp1:
    PRC2_GENESETS["NUYTTEN_NIPP1_TARGETS_UP"] = nuytten_nipp1

# Function to check overlap
def check_geneset_overlap(df, geneset_dict, fdr_cutoff=0.05):
    """
    Check overlap between DEGs and gene sets
    """
    results = []
    
    # Get significant DEGs
    sig_degs = df[df['FDR'] < fdr_cutoff].copy()
    sig_up = sig_degs[sig_degs['logFC'] > 0]
    sig_down = sig_degs[sig_degs['logFC'] < 0]

    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY:")
    print(f"Total significant DEGs (FDR < {fdr_cutoff}): {len(sig_degs)}")
    print(f"  - Upregulated: {len(sig_up)}")
    print(f"  - Downregulated: {len(sig_down)}")
    print(f"{'='*80}\n")
    
    for geneset_name, genes in geneset_dict.items():
        genes_set = set(genes)
        
        # Overall overlap
        overlap_all = sig_degs[sig_degs['gene'].isin(genes_set)]
        overlap_up = sig_up[sig_up['gene'].isin(genes_set)]
        overlap_down = sig_down[sig_down['gene'].isin(genes_set)]
        
        if len(overlap_all) > 0:
            print(f"\n{'='*80}")
            print(f"GENE SET: {geneset_name}")
            print(f"{'='*80}")
            print(f"Gene set size: {len(genes_set)}")
            print(f"Total overlap: {len(overlap_all)} genes ({len(overlap_all)/len(genes_set)*100:.1f}% of gene set)")
            print(f"  - Upregulated: {len(overlap_up)} genes")
            print(f"  - Downregulated: {len(overlap_down)} genes")
            
            # Show top hits by cell type
            if len(overlap_all) > 0:
                print(f"\nBreakdown by cell type:")
                for cell_type in overlap_all['cell_type'].unique():
                    ct_hits = overlap_all[overlap_all['cell_type'] == cell_type]
                    ct_up = len(ct_hits[ct_hits['logFC'] > 0])
                    ct_down = len(ct_hits[ct_hits['logFC'] < 0])
                    print(f"  {cell_type}: {len(ct_hits)} genes (↑{ct_up} ↓{ct_down})")
                
                print(f"\nTop genes by |logFC|:")
                tmp = overlap_all.copy()
                tmp["abs_logFC"] = tmp["logFC"].abs()
                top_genes = tmp.sort_values("abs_logFC", ascending=False).head(10)
                for _, row in top_genes.iterrows():
                    direction = "↑" if row['logFC'] > 0 else "↓"
                    print(f"  {direction} {row['gene']:12s} logFC={row['logFC']:6.2f} FDR={row['FDR']:.2e} ({row['cell_type']})")
            
            results.append({
                'geneset': geneset_name,
                'geneset_size': len(genes_set),
                'total_overlap': len(overlap_all),
                'overlap_pct': len(overlap_all)/len(genes_set)*100,
                'n_up': len(overlap_up),
                'n_down': len(overlap_down)
            })
    
    return pd.DataFrame(results)

# Run the analysis
print("\n" + "="*80)
print("CHECKING BC PATIENT-LEVEL DEGs FOR PRC2/H3K27me3 GENE SET ENRICHMENT")
print("="*80)

results_df = check_geneset_overlap(bc_patient, PRC2_GENESETS, fdr_cutoff=0.05)

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
if len(results_df) > 0:
    print(results_df.to_string(index=False))
else:
    print("NO SIGNIFICANT OVERLAPS FOUND")

# Check if PRC2 core components themselves are differentially expressed
print("\n" + "="*80)
print("PRC2 CORE COMPONENTS IN PATIENT-LEVEL DATA")
print("="*80)
prc2_core_check = bc_patient[bc_patient['gene'].isin(PRC2_GENESETS['PRC2_core'])]
if len(prc2_core_check) > 0:
    print(prc2_core_check[['gene', 'cell_type', 'logFC', 'FDR']].to_string(index=False))
else:
    print("No PRC2 core components found in patient-level DEGs")

# Also check the specific genes you showed (IL7R, FBLN2, MRPL12)
print("\n" + "="*80)
print("CHECKING YOUR EXAMPLE GENES (IL7R, FBLN2, MRPL12)")
print("="*80)
example_genes = ['IL7R', 'FBLN2', 'MRPL12']
example_check = bc_patient[bc_patient['gene'].isin(example_genes)]
if len(example_check) > 0:
    print(example_check[['gene', 'cell_type', 'logFC', 'logCPM', 'FDR']].to_string(index=False))
else:
    print("These genes not found in BC patient-level data")