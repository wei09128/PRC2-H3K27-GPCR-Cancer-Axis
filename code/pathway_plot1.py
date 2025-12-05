import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from typing import List

# --- CONFIGURATION ---
DEFAULT_FOLDER = '/mnt/f/H3K27/Data/CCC_BC/'
DEFAULT_N = 5
DEFAULT_TITLE = 'Top Enriched Pathways'
OUTPUT_DIR = DEFAULT_FOLDER + 'plot'

EXCLUDED_CT = {'Unassigned'}

def safe_filter_celltypes(df: pd.DataFrame, excluded, col='Cell_Type'):
    """Return df with excluded cell types removed, guarding empties/missing col."""
    if df is None or df.empty or col not in df.columns:
        return df
    return df[~df[col].isin(excluded)].copy()

def preprocess_gsea_data(df: pd.DataFrame, top_n: int, cell_type: str) -> pd.DataFrame:
    """Preprocess a single GSEA dataframe; sort by |NES| potency."""
    # Required cols
    required = ['FDR q-val', 'NES', 'Term']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  Error: Missing {missing} in {cell_type} GSEA.")
        print(f"  Available: {list(df.columns)}")
        return pd.DataFrame()

    # *** FIX: Filter by significance (FDR < 0.25) ***
    df = df[df['FDR q-val'] < 0.25].copy()
    if df.empty:
        print(f"  Warning: No significant pathways in {cell_type} GSEA (FDR < 0.25).")
        return pd.DataFrame()

    print(f"  → {cell_type}: {len(df)} pathways pass FDR < 0.25 threshold")

    # Metrics
    df['-log10(FDR)'] = -np.log10(df['FDR q-val'] + sys.float_info.min)

    # Extract gene count
    def extract_gene_count(tag_str):
        if pd.isna(tag_str): return 0
        s = str(tag_str).strip()
        if '/' in s:
            try: return int(s.split('/')[0])
            except: pass
        if '-' in s:
            parts = s.split('-')
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
        if '%' in s:
            try: return int(float(s.replace('%', '')))
            except: pass
        return 0

    if 'Tag %' in df.columns:
        df['Gene_Count'] = df['Tag %'].apply(extract_gene_count)
    elif 'Gene %' in df.columns:
        df['Gene_Count'] = df['Gene %'].apply(extract_gene_count)
    else:
        print("  Warning: No 'Tag %'/'Gene %' found. Using default Gene_Count=10.")
        df['Gene_Count'] = 10

    # Clean term
    df['Term_Clean'] = df['Term'].str.replace('_', ' ', regex=False)
    df['Term_Clean'] = df['Term_Clean'].apply(lambda x: x if len(x) < 50 else x[:47] + '...')

    # *** FIX: Potency = FDR q-val for sorting (lower FDR = higher significance) ***
    df['Potency'] = df['FDR q-val']
    df = df.sort_values(by='Potency', ascending=True).head(top_n).copy()  # ascending=True for lowest FDR first
    df['Cell_Type'] = cell_type
    
    # Filter excluded cell types
    df = safe_filter_celltypes(df, EXCLUDED_CT)
    
    print(f"    → Top {len(df)} pathways selected by FDR q-val (lowest/most significant)")
    return df

def combined_gsea_lollipop(
    all_gsea_data: List[pd.DataFrame],
    title: str,
    filename: str,
    top_n: int | None = None,
    mode: str = "per_cell",   # "per_cell" or "global"
    overall_cap: int = 80     # safety cap so the figure doesn't explode
):
    """
    Create a lollipop plot for GSEA results only (no ORA).
    
    Parameters:
    -----------
    all_gsea_data : List[pd.DataFrame]
        List of preprocessed GSEA dataframes
    title : str
        Plot title
    filename : str
        Output filename
    top_n : int, optional
        Number of top pathways per cell type (default: DEFAULT_N)
    mode : str
        "per_cell" = top-N per cell type, "global" = top-N overall
    overall_cap : int
        Maximum total pathways to plot
    """
    if not all_gsea_data:
        print("  Skipping GSEA lollipop — no data.")
        return

    # Concatenate all GSEA data
    g = pd.concat(
        [x for x in all_gsea_data if x is not None and not x.empty],
        ignore_index=True
    )
    if g.empty:
        print("  Skipping GSEA lollipop — all frames empty.")
        return

    # Ensure Potency = FDR q-val exists (lower = more significant)
    if 'Potency' not in g.columns and 'FDR q-val' in g.columns:
        g['Potency'] = g['FDR q-val']

    N = DEFAULT_N if top_n is None else int(top_n)

    # Select pathways based on mode
    if mode == "per_cell":
        # Top-N per cell type, then union
        parts = []
        for ct, sub in g.groupby('Cell_Type', sort=False):
            pick = sub.sort_values('Potency', ascending=True).head(N)  # ascending=True for lowest FDR
            parts.append(pick)
            print(f"  → {ct}: selected {len(pick)} pathways (FDR < 0.25, top by lowest FDR)")
        df_plot = pd.concat(parts, ignore_index=True)

        # Drop excluded cell types
        df_plot = safe_filter_celltypes(df_plot, EXCLUDED_CT)

        # Optional global cap
        if len(df_plot) > overall_cap:
            print(f"  → Capping at {overall_cap} pathways (was {len(df_plot)})")
            df_plot = (
                df_plot.sort_values('Potency', ascending=True)  # ascending=True for lowest FDR
                       .head(overall_cap)
            )
    else:
        # Global top-N across all cell types (by lowest FDR per term)
        top_terms = (
            g.groupby('Term_Clean')['Potency'].min()  # min = lowest FDR = most significant
             .sort_values(ascending=True)
             .head(N).index
        )
        df_plot = g[g['Term_Clean'].isin(top_terms)].copy()

    if df_plot.empty:
        print("  Nothing left after selection for GSEA lollipop.")
        return

    print(f"\n  Final plot contains {len(df_plot)} data points from {df_plot['Cell_Type'].nunique()} cell types")

    # Y-order: by lowest FDR (most significant) across selected rows
    order = (
        df_plot.groupby('Term_Clean')['Potency']
               .min().sort_values(ascending=True).index  # min FDR = most significant at top
    )
    df_plot['Term_Clean'] = pd.Categorical(
        df_plot['Term_Clean'],
        categories=order,
        ordered=True
    )
    df_plot = df_plot.sort_values(['Term_Clean', 'Cell_Type']).copy()

    # Fixed colors for each cell type
    CELLTYPE_COLOR_MAP = {
        'BC':              '#1f77b4',
        'Normal_Epithelial': '#ff7f0e',
        'Fibroblasts':     '#2ca02c',
        'Myofibroblasts':  '#98df8a',
        'Endothelial':     '#d62728',
        'Pericytes':       '#ff9896',
        'Adipocytes':      '#9467bd',
        'Mesothelial':     '#c5b0d5',
        'Macrophages':     '#8c564b',
        'Monocytes':       '#c49c94',
        'CD4_T':           '#e377c2',
        'CD8_T':           '#f7b6d2',
        'Tregs':           '#7f7f7f',
        'NK_Cells':        '#c7c7c7',
        'B_Cells':         '#bcbd22',
        'Plasma':          '#dbdb8d',
        'Dendritic':       '#17becf',
        'Mast':            '#9edae5',
        'Smooth_Muscle':   '#393b79',
    }

    # Get unique cell types and create color map
    cell_types = list(pd.unique(df_plot['Cell_Type']))
    fallback_palette = sns.color_palette("tab20", n_colors=len(cell_types))
    fallback_map = {ct: fallback_palette[i % len(fallback_palette)]
                    for i, ct in enumerate(cell_types)}

    color_map = {
        ct: CELLTYPE_COLOR_MAP.get(ct, fallback_map[ct])
        for ct in cell_types
    }

    # Calculate x-axis range
    max_abs = float(np.nanmax(np.abs(df_plot['NES']))) if len(df_plot) else 1.0
    max_abs = max(max_abs, 1.0)
    pad = 0.5

    # Create plot
    sns.set_style("whitegrid")
    height = max(6, 0.40 * len(order))
    fig, ax = plt.subplots(figsize=(12, height))

    # Draw stems (colored by cell type)
    for _, r in df_plot.iterrows():
        ax.plot(
            [0, r['NES']],
            [r['Term_Clean'], r['Term_Clean']],
            linewidth=1.0,
            alpha=0.45,
            color=color_map[r['Cell_Type']]
        )

    # Draw points (colored by cell type)
    for ct, sub in df_plot.groupby('Cell_Type', sort=False):
        # Bubble size ~ Gene_Count
        if 'Gene_Count' in sub.columns:
            sizes = np.clip(sub['Gene_Count'].fillna(1), 1, None) * 6
        else:
            sizes = np.full(len(sub), 60.0)

        ax.scatter(
            sub['NES'],
            sub['Term_Clean'],
            s=sizes,
            c=[color_map[ct]],
            alpha=0.9,
            edgecolors='black',
            linewidth=0.6,
            label=ct
        )

    # Formatting
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlim([-max_abs - pad, max_abs + pad])
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=12)
    ax.set_ylabel('Pathway Term (Gene Set)', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    ax.legend(
        title='Cell Type',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Combined GSEA lollipop saved: {filename}")

def run_batch_plotting(folder_path: str, top_n: int, overall_title: str):
    """Main function to discover GSEA files and create plot."""
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Searching for GSEA results in: {folder_path}")
    print(f"Outputs will be saved to: {OUTPUT_DIR}")
    print('='*60)

    # Discover GSEA files
    gsea_files: List[tuple[str, str]] = []
    gsea_sub_path = os.path.join(folder_path, 'GSEA_Prerank_Results')
    if os.path.isdir(gsea_sub_path):
        for filename in os.listdir(gsea_sub_path):
            if filename.endswith('.csv') and 'GSEA' in filename:
                cell_type = filename[:-4]
                for suf in ['_GSEA_Prerank_Results', '_GSEA_Results', '_GSEA_Prerank', '_GSEA']:
                    if cell_type.endswith(suf):
                        cell_type = cell_type[: -len(suf)]
                        break
                gsea_files.append((os.path.join(gsea_sub_path, filename), cell_type))
    else:
        print(f"  Error: GSEA_Prerank_Results directory not found at {gsea_sub_path}")
        sys.exit(1)

    print(f"\nFound {len(gsea_files)} GSEA files")

    # Process GSEA files
    all_gsea_data: List[pd.DataFrame] = []
    for file_path, cell_type in gsea_files:
        if cell_type in EXCLUDED_CT:
            print(f"  Skipping {cell_type} (in exclusion list)")
            continue
        try:
            print(f"\nProcessing {cell_type}...")
            df = pd.read_csv(file_path)
            df_plot = preprocess_gsea_data(df, top_n, cell_type)
            if df_plot is None or df_plot.empty:
                continue
            all_gsea_data.append(df_plot)
        except Exception as e:
            print(f"  Error processing GSEA for {cell_type}: {e}")
            import traceback
            traceback.print_exc()

    # Create combined plot
    print(f"\n{'='*60}")
    print(f"Creating combined GSEA lollipop plot")
    print(f"Total cell types with data: {len(all_gsea_data)}")
    print('='*60)

    combined_gsea_lollipop(
        all_gsea_data=all_gsea_data,
        title=f'{overall_title}: Combined GSEA (FDR < 0.25)',
        filename=os.path.join(OUTPUT_DIR, 'Combined_GSEA_Lollipop.png'),
        top_n=top_n,
        mode="per_cell"
    )

if __name__ == "__main__":
    FOLDER_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER
    try:
        N = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    except ValueError:
        print("Warning: N must be an integer. Using default.")
        N = DEFAULT_N
    TITLE = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_TITLE

    print(f"Starting GSEA plotting with N={N} per cell type...")
    run_batch_plotting(FOLDER_PATH, N, TITLE)
    print("\nBatch plotting complete.")