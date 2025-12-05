import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import logging
import seaborn as sns
import matplotlib.patches as mpatches
import sys
import anndata

# Configure logging
logging.basicConfig(level=logging.INFO)
plt.style.use('ggplot')

# ==============================================================================
# 0. USER INPUT DEFINITION
# ==============================================================================

H5AD_FILE_PATH = '/mnt/f/H3K27/Data/scRNA/scRNA_final.h5ad'
CELL_TYPE_COLUMN = 'cell_type'
CANCER_TYPE_COLUMN = 'Cancer_Type' # NEW: Column for cancer types

# Define immune cell types to analyze
IMMUNE_CELL_TYPES = [
    # 'BC','OC','EC','GC','Endothelial','Normal_Epithelial',
    # 'CD8_T', 'CD4_T',
    # 'B_Cells', 'Plasma',
    # 'NK_Cells',
    # 'Monocytes', 'Macrophages',
    # 'Dendritic', 'Tregs',
    # 'Mast',
    # 'Fibroblasts',
    'Myofibroblasts', 'Normal_Epithelial', 'Endothelial',
    # 'Pericytes','Smooth_Muscle', 'Adipocytes', 'Mesothelial',
    # 'Normal_Epithelial',
]

try:
    adata = sc.read_h5ad(H5AD_FILE_PATH)
    print(f"Successfully loaded data from {H5AD_FILE_PATH}.")
except FileNotFoundError:
    print(f"ERROR: File not found at {H5AD_FILE_PATH}. Please check the path.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading H5AD file: {e}")
    sys.exit(1)

# Verify required columns exist
if CELL_TYPE_COLUMN not in adata.obs.columns:
    print(f"ERROR: Column '{CELL_TYPE_COLUMN}' not found in adata.obs.")
    print(f"Available columns: {list(adata.obs.columns)}")
    sys.exit(1)

if CANCER_TYPE_COLUMN not in adata.obs.columns:
    print(f"ERROR: Column '{CANCER_TYPE_COLUMN}' not found in adata.obs.")
    print(f"Available columns: {list(adata.obs.columns)}")
    sys.exit(1)

# ==============================================================================
# 1. MARKER DEFINITION (Standardized to {'markers': [...]})
# ==============================================================================

immune_markers = {
    # Tumor-specific markers (often used as reference)
    'BC': {'markers': ['ESR1', 'ERBB2', 'GATA3', 'KRT19', 'FOXA1', 'PGR', 'IGHG1', 'IGHA1', 'IGKC','CD27', 'IGHD']},
    'OC': {'markers': ['MUC16', 'WFDC2', 'PAX8', 'WT1']},
    'EC': {'markers': ['ESR1', 'PGR', 'PAX8', 'PAEP']},
    'GC': {'markers': ['KIT', 'CD34', 'DOG1', 'PDGFRA']},

    # Stromal cells
    'Fibroblasts': {'markers': ['DCN', 'COL1A1', 'COL1A2','COL3A1', 'PDGFRA', 'LUM', 'FN1', 'FAP', 'PDPN', 'ACTA2', 'TAGLN']},
    'Myofibroblasts': {'markers': ['ACTA2', 'TAGLN', 'MYH11', 'PDGFRB', 'POSTN']},
    'Endothelial': {'markers': ['PECAM1', 'VWF', 'CDH5', 'CLDN5', 'FLT1', 'KDR']},
    'Pericytes': {'markers': ['RGS5', 'PDGFRB', 'NDUFA4L2', 'MCAM']},
    'Smooth_Muscle': {'markers': ['ACTA2', 'MYH11', 'TAGLN', 'CNN1', 'MYOCD']},
    'Normal_Epithelial': {'markers': ['EPCAM', 'KRT8', 'KRT18', 'CDH1', 'MUC1']},
    'Adipocytes': {'markers': ['ADIPOQ', 'PLIN1', 'FABP4', 'LPL']},
    'Mesothelial': {'markers': ['MSLN', 'UPK3B', 'KRT5', 'WT1', 'CALB2']},

    # Immune cells (Standardized to the dictionary format)
    # NOTE: The non-standard marker 'HAVCR2 (TIM-3)' is cleaned to 'HAVCR2'
    'CD8_T': {'markers': ['CD8A', 'CD8B', 'CD3D', 'CD3E', 'GZMB', 'PRF1', 'IFNG', 'HAVCR2']},
    'CD4_T': {'markers': ['CD4', 'CD3D', 'CD3E', 'IL7R', 'LTB', 'IL2', 'CCR7']},
    'Tregs': {'markers': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TIGIT', 'LAG3']},
    'NK_Cells': {'markers': ['NCAM1', 'NKG7', 'GNLY', 'KLRD1', 'KLRF1', 'FCGR3A','PDCD1','LAG3','TIGIT','HAVCR2']},
    'B_Cells': {'markers': ['CD79A', 'MS4A1', 'CD19', 'CD27', 'BANK1']},
    'Plasma': {'markers': ['IGHG1', 'IGHG3', 'JCHAIN', 'MZB1', 'SDC1', 'XBP1']},
    'Macrophages': {'markers': ['CD68', 'CD163', 'C1QA', 'C1QB', 'MSR1', 'PDL1', 'ARG1', 'MRC1', 'VEGFA', 'SEPP1', 'NOS2', 'IL1B']},
    'Monocytes': {'markers': ['CD14', 'FCGR3A', 'S100A8', 'S100A9', 'LYZ', 'HLA-DRA']},
    'Dendritic': {'markers': ['CD1C', 'CLEC9A', 'FCER1A', 'CLEC10A', 'CCR7', 'CD83']},
    'Mast': {'markers': ['TPSAB1', 'TPSB2', 'CPA3', 'HDC', 'KIT']},
    
    
    # 3. Inflammatory CAFs (iCAF) - Immune regulation via cytokine and chemokine secretion
    'Inflammatory CAFs': {
        'markers': [
            'IL6',      # Interleukin 6
            'LIF',      # Leukemia Inhibitory Factor
            'CXCL12',   # Chemokine (C-X-C motif) Ligand 12
            'IL1R1',    # Interleukin 1 Receptor Type 1
            'CXCL1',    # Chemokine (C-X-C motif) Ligand 1
            'CXCL14',   # Chemokine (C-X-C motif) Ligand 14
            'PTGS2'     # Prostaglandin-Endoperoxide Synthase 2 (COX-2)
        ]
    },

    # 4. Antigen-Presenting CAFs (apCAF) - Immune modulation and antigen presentation
    'Antigen-Presenting CAFs': {
        'markers': [
            'HLA-DRA',  # MHC Class II
            'HLA-DRB1', # MHC Class II
            'CD74',     # Invariant chain
            'CIITA',    # MHC Class II Transcriptional Activator
            'CD40'      # Tumor necrosis factor receptor superfamily member 5
        ]
    },
}

# ==============================================================================
# 2. FILTER TO IMMUNE CELLS AND VALIDATE
# ==============================================================================

# Filter to only immune cells
adata_immune = adata[adata.obs[CELL_TYPE_COLUMN].isin(IMMUNE_CELL_TYPES)].copy()

if adata_immune.n_obs == 0:
    print("ERROR: No immune cells found in the dataset.")
    sys.exit(1)

print(f"\nFound {adata_immune.n_obs} immune cells across {len(adata_immune.obs[CANCER_TYPE_COLUMN].unique())} cancer types")
print(f"Cancer types: {sorted(adata_immune.obs[CANCER_TYPE_COLUMN].unique())}")
print(f"Immune cell types: {sorted(adata_immune.obs[CELL_TYPE_COLUMN].unique())}")

# Get valid immune cell types and cancer types present in data
valid_immune_types = [ct for ct in IMMUNE_CELL_TYPES if ct in adata_immune.obs[CELL_TYPE_COLUMN].unique()]
cancer_types = sorted(adata_immune.obs[CANCER_TYPE_COLUMN].unique())

# Build gene order with groups
gene_order = []
gene_groups = []
seen_genes = set()

for cell_type in valid_immune_types:
    if cell_type in immune_markers:
        # **CORRECTION**: Access the list of markers via the 'markers' key
        # We assume the standardized dictionary structure: {'markers': [...]}.
        marker_list = immune_markers[cell_type].get('markers', [])

        # Filter and clean markers against the actual genes in the data
        for gene_with_suffix in marker_list:
            # Clean up non-standard marker names (e.g., 'HAVCR2 (TIM-3)')
            gene = gene_with_suffix.split(' ')[0].split('(')[0].strip()

            if gene in adata_immune.var_names:
                if gene not in seen_genes:
                    gene_order.append(gene)
                    gene_groups.append(cell_type)
                    seen_genes.add(gene)

if not gene_order:
    # This error check will now correctly report if no *actual genes* were found
    print("ERROR: No valid marker genes found in the dataset after filtering against adata.var_names.")
    sys.exit(1)

# ==============================================================================
# 3. CREATE COMBINED GROUPING (Cell_Type + Cancer_Type)
# ==============================================================================

# Create combined label with conditional logic
adata_immune.obs['CellType_CancerType'] = [
    ct if ct == cancer else f"{ct} in {cancer}"
    for ct, cancer in zip(
        adata_immune.obs[CELL_TYPE_COLUMN].astype(str),
        adata_immune.obs[CANCER_TYPE_COLUMN].astype(str)
    )
]

# Create ordered categories: for each immune cell type, show all cancer types
combined_order = []
# Ensure unique labels are used, as the set of labels in the column is the gold standard
unique_labels = adata_immune.obs['CellType_CancerType'].unique()

for immune_type in valid_immune_types:
    # 1. Add the "type-matched" label first
    type_matched_label = immune_type # e.g., 'BC'
    if type_matched_label in unique_labels and type_matched_label not in combined_order:
        combined_order.append(type_matched_label)
        
    # 2. Add the "cross-sample" labels
    for cancer_type in cancer_types:
        # Skip the type-matched case as it's already added or invalid
        if immune_type == cancer_type:
            continue
            
        # The label format is 'CellType in CancerType sample'
        cross_sample_label = f"{immune_type} in {cancer_type}"
        
        if cross_sample_label in unique_labels and cross_sample_label not in combined_order:
            combined_order.append(cross_sample_label)

# Set categorical order
if len(combined_order) > 0:
    adata_immune.obs['CellType_CancerType'] = (
        adata_immune.obs['CellType_CancerType']
        .astype('category')
        .cat.set_categories(combined_order)
    )
    print(f"Set {len(combined_order)} ordered categories.") # Add a check here
else:
    print("ERROR: Combined grouping resulted in no valid categories.")
    sys.exit(1)


# ==============================================================================
# 4. CREATE ROW COLORS FOR GENE GROUPS
# ==============================================================================

cluster_colors = plt.cm.get_cmap('tab10', len(valid_immune_types))
color_map = {ct: cluster_colors(i) for i, ct in enumerate(valid_immune_types)}
row_colors_series = pd.Series([color_map[g] for g in gene_groups], index=gene_order)

# ==============================================================================
# 5. PLOT HEATMAP
# ==============================================================================

print("\nGenerating heatmap comparing immune cells across cancer types...")

fig = plt.figure(figsize=(max(12, len(combined_order) * 0.3), len(gene_order) * 0.35 + 2))

sc.pl.heatmap(
    adata_immune,
    var_names=gene_order,
    groupby='CellType_CancerType',
    dendrogram=False,
    standard_scale='var',
    cmap='RdBu_r',
    swap_axes=True,
    figsize=(max(12, len(combined_order) * 0.3), len(gene_order) * 0.35 + 2),
    show_gene_labels=True,
    show=False
)

# Rotate labels (works regardless of swap_axes)
for ax in plt.gcf().axes:
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    for label in ax.get_xticklabels():
        label.set_ha('right')

plt.suptitle('Immune Cell Marker Expression Across Cancer Types (Z-score)',
             fontsize=14, y=0.985)

# Add legend for immune cell types
handles = [mpatches.Patch(color=color_map[ct], label=ct) for ct in valid_immune_types]
plt.legend(
    handles=handles,
    title="Immune Cell Type",
    bbox_to_anchor=(1.02, 0.5),
    loc='center left',
    borderaxespad=0.,
    fontsize=8,
    title_fontsize=10,
)

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig('immune_cells_by_cancer_type_heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved as 'immune_cells_by_cancer_type_heatmap.png'")

# ==============================================================================
# 6. OPTIONAL: STATISTICAL ANALYSIS
# ==============================================================================

print("\n" + "="*60)
print("IMMUNE CELL DISTRIBUTION ACROSS CANCER TYPES")
print("="*60)

# Calculate cell counts
cell_counts = adata_immune.obs.groupby([CELL_TYPE_COLUMN, CANCER_TYPE_COLUMN]).size().unstack(fill_value=0)
print("\nCell counts:")
print(cell_counts)

# Calculate proportions
cell_props = cell_counts.div(cell_counts.sum(axis=0), axis=1) * 100
print("\nProportions (%):")
print(cell_props.round(2))

# Save statistics
cell_counts.to_csv('immune_cell_counts_by_cancer.csv')
cell_props.to_csv('immune_cell_proportions_by_cancer.csv')
print("\nStatistics saved to CSV files.")

plt.show()