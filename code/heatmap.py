import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
import sys

# Configure logging to suppress verbose output from Scanpy/Matplotlib
logging.basicConfig(level=logging.INFO)
plt.style.use('ggplot')

# ==============================================================================
# 0. INPUT DEFINITION
# ==============================================================================

H5AD_FILE_PATH = '/mnt/f/H3K27/Data/scRNA/scRNA_final.h5ad'
CELL_TYPE_COLUMN = 'cell_type' # Ensure this matches the column in adata.obs

try:
    # 1. Load the AnnData object from the H5AD file
    adata = sc.read_h5ad(H5AD_FILE_PATH)
    print(f"Successfully loaded data from {H5AD_FILE_PATH}.")
except FileNotFoundError:
    print(f"ERROR: File not found at {H5AD_FILE_PATH}. Please check the path.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading H5AD file: {e}")
    sys.exit(1)

# Ensure the required column for clustering is present
if CELL_TYPE_COLUMN not in adata.obs.columns:
    print(f"ERROR: Column '{CELL_TYPE_COLUMN}' not found in adata.obs.")
    print(f"Available columns: {list(adata.obs.columns)}")
    sys.exit(1)


# 2. Define gene groupings and cluster order based on the provided data structure
yaml_data = {'cell_types': {
    'BC': {'markers': ['ESR1', 'ERBB2', 'GATA3', 'KRT19', 'FOXA1', 'PGR']},
    'OC': {'markers': ['MUC16', 'WFDC2', 'PAX8', 'WT1']},
    'EC': {'markers': ['ESR1', 'PGR', 'PAX8', 'PAEP']},
    'GC': {'markers': ['KIT', 'CD34', 'DOG1', 'PDGFRA']},
    'CD8_T': {'markers': ['CD8A', 'CD8B', 'CD3D', 'CD3E', 'GZMB']},
    'CD4_T': {'markers': ['CD4', 'CD3D', 'CD3E', 'IL7R', 'LTB']},
    'Tregs': {'markers': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2']},
    'NK_Cells': {'markers': ['NCAM1', 'NKG7', 'GNLY', 'KLRD1', 'KLRF1']},
    'B_Cells': {'markers': ['CD79A', 'MS4A1', 'CD19']},
    'Plasma': {'markers': ['IGHG1', 'IGHG3', 'JCHAIN', 'MZB1', 'SDC1']},
    'Macrophages': {'markers': ['CD68', 'CD163', 'C1QA', 'C1QB', 'MSR1']},
    'Monocytes': {'markers': ['CD14', 'FCGR3A', 'S100A8', 'S100A9', 'LYZ']},
    'Dendritic': {'markers': ['CD1C', 'CLEC9A', 'FCER1A', 'CLEC10A']},
    'Mast': {'markers': ['TPSAB1', 'TPSB2', 'CPA3', 'HDC']},
    'Fibroblasts': {'markers': ['DCN', 'COL1A1', 'COL1A2', 'PDGFRA', 'LUM', 'FN1']},
    'Myofibroblasts': {'markers': ['ACTA2', 'TAGLN', 'MYH11', 'PDGFRB']},
    'Endothelial': {'markers': ['PECAM1', 'VWF', 'CDH5', 'CLDN5']},
    'Pericytes': {'markers': ['RGS5', 'PDGFRB', 'NDUFA4L2', 'MCAM']},
    'Smooth_Muscle': {'markers': ['ACTA2', 'MYH11', 'TAGLN', 'CNN1', 'MYOCD']},
    'Normal_Epithelial': {'markers': ['EPCAM', 'KRT8', 'KRT18', 'CDH1', 'MUC1']},
    'Adipocytes': {'markers': ['ADIPOQ', 'PLIN1', 'FABP4', 'LPL']},
    'Mesothelial': {'markers': ['MSLN', 'UPK3B', 'KRT5', 'WT1', 'CALB2']}
}}

cell_type_order_preference = [
    'BC', 'OC', 'EC', 'GC', 'CD8_T', 'CD4_T', 'Tregs', 'NK_Cells', 'B_Cells', 'Plasma',
    'Dendritic', 'Monocytes', 'Macrophages', 'Mast', 'Fibroblasts', 'Myofibroblasts',
    'Endothelial', 'Pericytes', 'Smooth_Muscle', 'Normal_Epithelial', 'Adipocytes', 'Mesothelial'
]

# Get the valid cluster names present in the data, following the preferred order
present_clusters = adata.obs[CELL_TYPE_COLUMN].unique().tolist()
valid_order = [ct for ct in cell_type_order_preference if ct in present_clusters]
other_categories = [ct for ct in present_clusters if ct not in valid_order]
final_cluster_order = valid_order + other_categories

if not valid_order:
    print(f"ERROR: None of the defined cell types were found in the '{CELL_TYPE_COLUMN}' column.")
    sys.exit(1)

# Reorder the cell type categories for plotting (X-axis order)
adata.obs[CELL_TYPE_COLUMN] = adata.obs[CELL_TYPE_COLUMN].astype('category').cat.reorder_categories(final_cluster_order)

# 3. Construct the ordered list of genes and their groups for the Y-axis
gene_order_for_heatmap = []
gene_groups = []
seen_genes = set()

for cell_type in valid_order:
    if cell_type in yaml_data['cell_types']:
        # Filter markers to only include those present in the AnnData object
        markers = [g for g in yaml_data['cell_types'][cell_type]['markers'] if g in adata.var_names]

        # Add genes that haven't been added yet
        for gene in markers:
            if gene not in seen_genes:
                gene_order_for_heatmap.append(gene)
                gene_groups.append(cell_type)
                seen_genes.add(gene)

if not gene_order_for_heatmap:
    print("ERROR: None of the defined marker genes were found in the dataset.")
    sys.exit(1)


# 4. Define colors for gene groups (for the side bar)
# Create a color map that maps each cell group name to a distinct color
cluster_colors = plt.cm.get_cmap('tab20', len(valid_order))
color_map = {ct: cluster_colors(i) for i, ct in enumerate(valid_order)}


# ==============================================================================
# 5. PLOT THE HEATMAP
# ==============================================================================
print("Generating Z-score heatmap...")

# Plot heatmap and capture the axes dictionary
axes_dict = sc.pl.heatmap( 
    adata,
    var_names=gene_order_for_heatmap,
    groupby=CELL_TYPE_COLUMN,
    dendrogram=False,
    standard_scale='var',
    cmap='RdBu_r',
    swap_axes=True,
    figsize=(14, 20),
    show_gene_labels=True,
    show=False
)

# --- FIX APPLIED HERE ---
# Access the main heatmap axis and apply the rotation to the x-axis labels.
if 'heatmap_ax' in axes_dict:
    axes_dict['heatmap_ax'].tick_params(axis='x', rotation=45, labelrotation=45)
else:
    # Fallback for unexpected Scanpy versions/outputs, often targeting the whole figure
    plt.xticks(rotation=45, ha='right')
    print("Warning: Could not find 'heatmap_ax'. Applying global rotation as fallback.")


# Clean up default color strip labels
for ax in plt.gcf().axes:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        
# Adjust suptitle position
plt.suptitle('Marker Gene Expression (Z-score) Across Cell Types', fontsize=16, y=0.985)

# Add a clean legend for gene groups
handles = [mpatches.Patch(color=color_map[ct], label=ct) for ct in valid_order]

plt.legend(
    handles=handles,
    title="Cell Type (Gene Group)",
    bbox_to_anchor=(1.02, 0.5),
    loc='center left',
    borderaxespad=0.,
    fontsize=8,
    title_fontsize=10,
)

plt.tight_layout(rect=[0, 0, 0.85, 1]) # leave space for legend
plt.savefig('marker_gene_heatmap.png', dpi=300)
print("Heatmap saved as 'marker_gene_heatmap.png'")

# plt.show()