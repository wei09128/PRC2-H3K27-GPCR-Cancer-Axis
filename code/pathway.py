#!/usr/bin/env python3
import pandas as pd
import gseapy as gp
import os
import numpy as np
import sys
import argparse
from typing import Dict, Any

# ==============================================================================
# 0. CONFIGURATION (DEFAULTS)
# ==============================================================================

# GMT file (still hard-coded here, but could also be made an argument if needed)
GMT_FILE_PATH = '/mnt/f/H3K27/Data/CCC/c2.all.v2023.1.Hs.symbols.gmt'

# These are DEFAULTS (can be overridden by CLI arguments)
DEFAULT_ORA_INPUT_PATH = '/mnt/f/H3K27/Data/CCC_EC/dge_genes_for_pathway_analysis.csv'
DEFAULT_GSEA_INPUT_PATH = '/mnt/f/H3K27/Data/CCC_EC/dge_wilcoxon_full_results.csv'
DEFAULT_OUTPUT_DIR = '/mnt/f/H3K27/Data/CCC_EC/'

# Cell types to analyze (ensure these match the column headers in your CSVs)
CELL_TYPES = [
    'BC', 'OC', 'EC',
    'Fibroblasts','Endothelial','Myofibroblasts','Pericytes','Smooth_Muscle','Adipocytes',
    'Monocytes','Macrophages','CD4_T','CD8_T','Tregs','B_Cells','Plasma','NK_Cells','Mast',
    'Normal_Epithelial','Mesothelial','Unassigned'
]

# ==============================================================================
# 1. OVER-REPRESENTATION ANALYSIS (ORA) using gseapy.enrichr
# ==============================================================================

def perform_ora_analysis(genes_df: pd.DataFrame, gmt_file: str, outdir: str) -> Dict[str, Any]:
    """Performs ORA using gseapy.enrichr for each cell type's gene list."""
    print("--- Starting ORA (Enrichr) Analysis ---")
    ora_results = {}
    
    # Ensure the output subdirectory exists
    ora_outdir = os.path.join(outdir, 'ORA_Results')
    os.makedirs(ora_outdir, exist_ok=True)
    
    for ct in CELL_TYPES:
        if ct not in genes_df.columns:
            print(f"Skipping ORA for {ct}: Column not found in input file.")
            continue

        # Filter out empty/NaN entries and create a single list of unique genes
        gene_list = genes_df[ct].dropna().unique().tolist()
        
        if not gene_list:
            print(f"Skipping ORA for {ct}: No genes found in list.")
            continue
            
        print(f"Running ORA for {ct} with {len(gene_list)} genes...")
        
        # Run Enrichr using the local GMT file
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gmt_file,
            organism='human',
            outdir=None,  # Do not save files automatically, we handle saving below
            cutoff=0.05,  # Only report pathways with adjusted p-value < 0.05
            verbose=False,
        )

        # Save results to a clean CSV
        result_path = os.path.join(ora_outdir, f'{ct}_ORA_Enrichr_Results.csv')
        enr.results.to_csv(result_path, index=False)
        ora_results[ct] = enr.results
        print(f"  ✅ ORA results saved for {ct} to {result_path}")
        
    print("--- ORA Analysis Complete ---")
    return ora_results


# ==============================================================================
# 2. GENE SET ENRICHMENT ANALYSIS (GSEA) using gseapy.prerank
# ==============================================================================

def perform_gsea_prerank(full_dge_df: pd.DataFrame, gmt_file: str, outdir: str,
                         cell_types=None) -> Dict[str, Any]:
    print("\n--- Starting GSEA (Prerank) Analysis ---")
    gsea_results = {}

    gsea_outdir = os.path.join(outdir, 'GSEA_Prerank_Results')
    os.makedirs(gsea_outdir, exist_ok=True)

    # If not provided, infer from the data
    if cell_types is None:
        cell_types = sorted(full_dge_df['cell_type'].unique())

    for ct in cell_types:
        print(f"Preparing GSEA input for {ct}...")

        ct_df = full_dge_df[full_dge_df['cell_type'] == ct].copy()

        if ct_df.empty:
            print(f"Skipping GSEA for {ct}: No full DGE data found.")
            continue

        ct_df['rank_score'] = -np.log10(ct_df['pvals_adj'] + sys.float_info.min) * np.sign(ct_df['logfoldchanges'])

        ranked_series = ct_df.set_index('names')['rank_score'].sort_values(ascending=False)

        print(f"Running GSEA Prerank for {ct} with {len(ranked_series)} genes...")

        pre_res = gp.prerank(
            rnk=ranked_series,
            gene_sets=gmt_file,
            outdir=None,
            seed=42,
            format='pdf',
            verbose=False,
        )

        result_path = os.path.join(gsea_outdir, f'{ct}_GSEA_Prerank_Results.csv')
        pre_res.res2d.to_csv(result_path, index=False)
        gsea_results[ct] = pre_res.res2d

        print(f"  ✅ GSEA Prerank results saved for {ct} to {result_path}")

    print("--- GSEA Analysis Complete ---")
    return gsea_results


def main():
    # ------------------------------------------------------------------
    # Parse CLI arguments for the 3 paths you want to pass from outside
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Run ORA and GSEA pathway analysis for DGE results."
    )
    parser.add_argument(
        "--ora_input",
        type=str,
        default=DEFAULT_ORA_INPUT_PATH,
        help=f"Path to ORA input CSV (top genes per cell type). Default: {DEFAULT_ORA_INPUT_PATH}"
    )
    parser.add_argument(
        "--gsea_input",
        type=str,
        default=DEFAULT_GSEA_INPUT_PATH,
        help=f"Path to GSEA input CSV (full DGE results). Default: {DEFAULT_GSEA_INPUT_PATH}"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for all results. Default: {DEFAULT_OUTPUT_DIR}"
    )
    parser.add_argument(
        "--cancer_type",
        type=str,
        default=None,
        help="Optional: filter GSEA input to a single cancer type using the 'Cancer_Type' column (e.g. BC, OC, EC)."
    )


    args = parser.parse_args()

    ora_input_path = args.ora_input
    gsea_input_path = args.gsea_input
    output_dir = args.outdir
    cancer_type = args.cancer_type

    # ------------------------------------------------------------------
    # Run with the provided paths
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using GMT file: {GMT_FILE_PATH}")
    print(f"ORA input:  {ora_input_path}")
    print(f"GSEA input: {gsea_input_path}")
    print(f"Results will be saved in: {output_dir}")

    # --- Load ORA Input (Filtered DEGs) ---
    try:
        ora_input_df = pd.read_csv(ora_input_path)
        print(f"Successfully loaded ORA input from {ora_input_path}")
    except FileNotFoundError:
        print(f"ERROR: ORA input file not found at {ora_input_path}. Creating a mock file structure.")
        ora_input_df = pd.DataFrame({
            'Breast_Cancer': ['ESR1', 'GATA3', 'JUN', 'FOS', 'IL6', 'MMP9', 'COL1A1', 'TAGLN', np.nan, np.nan],
            'Fibroblasts': ['DCN', 'LUM', 'TGFB1', 'PDGFRA', 'FAP', 'MMP14', 'IL6', 'CXCL12', np.nan, np.nan],
            'Ovarian_Cancer': ['PAX8', 'WT1', 'CDH1', 'MYC', 'VEGFA', 'CD44', 'FN1', 'POSTN', np.nan, np.nan],
            'Gastric_Cancer': ['KIT', 'CD34', 'DOG1', 'SOX2', 'CDX2', 'EPCAM', 'ACTA2', 'TFF3', np.nan, np.nan],
        })

    # --- Load GSEA Input (Full DGE Results) ---
    try:
        gsea_input_df = pd.read_csv(gsea_input_path)
        print(f"Successfully loaded GSEA input from {gsea_input_path}")
    except FileNotFoundError:
        print(f"ERROR: GSEA input file not found at {gsea_input_path}. Creating a mock file structure.")
        np.random.seed(42)
        mock_genes = [f'GENE_{i}' for i in range(1, 1000)]
        mock_data = []
        for ct in CELL_TYPES:
            df = pd.DataFrame({
                'names': mock_genes + [f'CT_GENE_{i}' for i in range(1000, 1100)],
                'logfoldchanges': np.random.uniform(-1, 1, 1100),
                'pvals_adj': 10**(-np.random.uniform(0.1, 5, 1100)),
                'cell_type': ct
            })
            mock_data.append(df)
        gsea_input_df = pd.concat(mock_data)
        
    if cancer_type is not None:
        if 'Cancer_Type' not in gsea_input_df.columns:
            print(f"WARNING: '--cancer_type {cancer_type}' was given, "
                  f"but 'Cancer_Type' column not found in {gsea_input_path}. No filtering was applied.")
        else:
            before = len(gsea_input_df)
            gsea_input_df = gsea_input_df[gsea_input_df['Cancer_Type'] == cancer_type].copy()
            after = len(gsea_input_df)
            print(f"Filtered GSEA input to Cancer_Type == {cancer_type}: {after} of {before} rows kept.")
    # ----------------------------------------------------
    # EXECUTE ANALYSES
    # ----------------------------------------------------
    ora_results = perform_ora_analysis(ora_input_df, GMT_FILE_PATH, output_dir)
    gsea_results = perform_gsea_prerank(gsea_input_df, GMT_FILE_PATH, output_dir)

    # ----------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)

    print("\nORA (Enrichr) Summary:")
    for ct, res in ora_results.items():
        print(f"  {ct}: Found {len(res)} significant pathways (Adjusted P < 0.05).")

    print("\nGSEA (Prerank) Summary:")
    for ct, res in gsea_results.items():
        print(f"  {ct}: Found {len(res[res['FDR q-val'] < 0.25])} pathways with FDR q-val < 0.25 (GSEA standard).")


if __name__ == "__main__":
    main()
