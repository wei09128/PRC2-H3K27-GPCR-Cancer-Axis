#!/usr/bin/env python3
"""
Build patient-level PRC2/H3K27 signature table from scRNA h5ad.

Input:
    scRNA_final.h5ad (cells x genes) with at least:
        adata.obs['Sample_ID']
        adata.obs['Cancer_Type']
        adata.obs['Response_Status']
        adata.obs['cell_type']

Output:
    CSV with one row per Sample_ID and columns:
        Sample_ID
        Cancer_Type
        Response_Status
        Aggressive_Label
        <celltype>__<signature_name>   (mean score per patient, per cell type)
"""

import sys
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import gseapy as gp
from gseapy.parser import read_gmt   # <-- add this
from sklearn.base import BaseEstimator, TransformerMixin

gmt_path = "/mnt/f/H3K27/Data/CCC/c2.all.v2023.1.Hs.symbols.gmt"
gmt_dict = read_gmt(gmt_path)       # <-- change this line

# ========================
# 1. USER-EDITABLE SETTINGS
# ========================

# (A) PRC2 / H3K27 gene sets
# ------------------------------------------------------------------
# IMPORTANT:
#   Replace these with the exact PRC2/H3K27 signatures you are using
#   (e.g. KONDO_PROSTATE_CANCER_WITH_H3K27ME3, MIKKELSEN_iPS_HCP_H3K27ME3).
#
#   Each entry is: "short_name": [list of gene symbols]
#
kondo_full      = gmt_dict["KONDO_PROSTATE_CANCER_WITH_H3K27ME3"]
mikkelsen_mef   = gmt_dict["MIKKELSEN_MEF_HCP_WITH_H3K27ME3"]
mikkelsen_mcv6  = gmt_dict["MIKKELSEN_MCV6_HCP_WITH_H3K27ME3"]
benporath_es    = gmt_dict["BENPORATH_ES_WITH_H3K27ME3"]
nuytten_ezh2    = gmt_dict["NUYTTEN_EZH2_TARGETS_UP"]
nuytten_nipp1   = gmt_dict["NUYTTEN_NIPP1_TARGETS_UP"]

PRC2_GENESETS_ORIGINAL = {
    "PRC2_core": [
        "EZH2", "EED", "SUZ12", "RBBP4", "RBBP7"
    ],

    # External H3K27/PRC2 sets only
    "KONDO_H3K27ME3": kondo_full,
    "MIKKELSEN_MEF_H3K27ME3": mikkelsen_mef,
    "MIKKELSEN_MCV6_H3K27ME3": mikkelsen_mcv6,
    "BENPORATH_ES_H3K27ME3": benporath_es,
    "NUYTTEN_EZH2_TARGETS_UP": nuytten_ezh2,
    "NUYTTEN_NIPP1_TARGETS_UP": nuytten_nipp1,
}

reduction_order = [
    "PRC2_core",
    "KONDO_H3K27ME3",
    "MIKKELSEN_MEF_H3K27ME3",
    "MIKKELSEN_MCV6_H3K27ME3",
    "BENPORATH_ES_H3K27ME3",
    "NUYTTEN_EZH2_TARGETS_UP",
    "NUYTTEN_NIPP1_TARGETS_UP",
]

# # Initialize an empty set to store unique genes
# unique_genes_set = set()
# # Iterate through the values of the dictionary
# for geneset in PRC2_GENESETS.values():
#     # Only process values that are lists (explicitly defined gene lists)
#     if isinstance(geneset, list):
#         unique_genes_set.update(geneset)
# # Convert the set back to a sorted list
# unique_genes_list = sorted(list(unique_genes_set))
# print(unique_genes_list)

# pick which Pan-Gyn list to use as reference
# pan_core = set(PRC2_GENESETS["PanGyn_top65_core"])
# print("=== Overlap with PanGyn_top65_core ===")
# for sig_name, genes in PRC2_GENESETS.items():
#     # skip the Pan-Gyn sets themselves
#     if sig_name.startswith("PanGyn"):
#         continue
#     overlap = sorted(set(genes) & pan_core)
#     print(f"{sig_name}: {len(overlap)} overlapping genes")
#     if overlap:
#         print("  ", ", ".join(overlap[:]))
#         if len(overlap) > 20:
#             print("   ...")
#     print("-" * 60)


# (B) Cell types of interest
# ------------------------------------------------------------------
# If None, the script will keep *all* cell types present in adata.obs["cell_type"].
# Otherwise, restrict to this list.
CELL_TYPES_OF_INTEREST = [
    "Fibroblasts",
    "Myofibroblasts",
    "Macrophages",
    "Monocytes",
    "CD4_T",
    "CD8_T",
    "Tregs",
    "NK_Cells",
    "Endothelial",
    "Normal_Epithelial",
    "BC",
    "OC",
    "EC",
    # add/remove as needed
]

def sequentially_reduce_genesets(original_genesets: dict, reduction_order: list) -> dict:
    """
    Performs a sequential subtraction of gene lists in the provided dictionary.

    Each gene set in the reduction_order list is made unique relative to 
    all gene sets that came before it in the list.

    Args:
        original_genesets (dict): The dictionary of gene sets where keys are 
                                  set names and values are lists of gene symbols (or placeholders).
        reduction_order (list): The list of keys specifying the exact order in which
                                subtraction should occur.

    Returns:
        dict: A new dictionary where each gene list is unique relative to its 
              preceding lists in the defined order.
    """
    genes_seen = set()
    reduced_genesets = {}

    # 1. Iterate through the keys in the defined order
    for key in reduction_order:
        if key not in original_genesets:
            print(f"Warning: Key '{key}' not found in original_genesets. Skipping.")
            continue

        geneset_value = original_genesets[key]

        # 2. Check if the value is a list (an actual gene set)
        if isinstance(geneset_value, list):
            # Convert the current list to a set for efficient subtraction
            current_geneset_set = set(geneset_value)

            # Subtract all genes seen so far (the unique reduction step)
            reduced_geneset = current_geneset_set - genes_seen

            # Store the result (sorted for consistency)
            reduced_genesets[key] = sorted(list(reduced_geneset))

            # Update the running set of ALL genes encountered so far using the ORIGINAL set
            genes_seen.update(current_geneset_set)
        
        else:
            # 3. Handle external/placeholder variables by copying them directly
            reduced_genesets[key] = geneset_value
            
    # 4. Add any remaining keys that were not in the reduction_order (e.g., if you only included 
    #    PanGyn sets in the order but not the external H3K27 sets)
    for key, value in original_genesets.items():
        if key not in reduced_genesets:
            reduced_genesets[key] = value

    return reduced_genesets
# ========================
# 2. HELPER FUNCTIONS
# ========================

def compute_module_score(adata, gene_list, use_raw_preferred=True):
    """
    Compute a simple module score: mean expression across gene_list for each cell.
    - First tries adata.var_names
    - If 0 genes found and raw is available, tries adata.raw.var_names
    - If still 0, returns zeros and prints a warning instead of raising.
    """
    import numpy as np
    from scipy import sparse

    if len(gene_list) == 0:
        # nothing defined for this signature
        return np.zeros(adata.n_obs), [], []

    # --- try main matrix ---
    var_names = adata.var_names
    genes_present = [g for g in gene_list if g in var_names]
    source = "adata"

    # --- if nothing, try raw if available ---
    if len(genes_present) == 0 and adata.raw is not None and use_raw_preferred:
        raw_var_names = adata.raw.var_names
        genes_present = [g for g in gene_list if g in raw_var_names]
        if len(genes_present) > 0:
            source = "raw"

    missing_genes = [g for g in gene_list if g not in genes_present]

    if len(genes_present) == 0:
        print("  0 genes found for this signature in this AnnData; skipping and returning zeros.")
        return np.zeros(adata.n_obs), [], gene_list

    if source == "adata":
        X_sub = adata[:, genes_present].X
    else:
        X_sub = adata.raw[:, genes_present].X

    if sparse.issparse(X_sub):
        X_sub = X_sub.toarray()

    scores = np.asarray(X_sub.mean(axis=1)).ravel()
    print(f"  {len(genes_present)} genes found, {len(missing_genes)} missing")
    return scores, genes_present, missing_genes



def map_response_status(status):
    """
    Map the detailed Response_Status to a coarser Aggressive_Label.

    - Aggressive / Persister / Recurrent / Residual / Resistant -> 'Aggressive'
    - less-Aggressive                                         -> 'LessAggressive'
    - non-Aggressive                                          -> 'NonAggressive'
    - anything else                                           -> 'Unknown'
    """
    aggressive_like = {"Aggressive", "Persister", "Recurrent", "Residual", "Resistant"}

    if status in aggressive_like:
        return "Aggressive"
    elif status == "less-Aggressive":
        return "LessAggressive"
    elif status == "non-Aggressive":
        return "NonAggressive"
    else:
        return "Unknown"


# ========================
# 3. MAIN
# ========================

def main(h5ad_path, out_csv_path):
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(f"Input h5ad not found: {h5ad_path}")

    print(f"Loading AnnData from: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    print(f"Loaded AnnData with shape: {adata.n_obs} cells x {adata.n_vars} genes")

    # Check required obs columns
    required_obs_cols = ["Sample_ID", "Cancer_Type", "Response_Status", "cell_type"]
    missing_cols = [c for c in required_obs_cols if c not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in adata.obs: {missing_cols}")

    # Optionally restrict to selected cell types
    if CELL_TYPES_OF_INTEREST is not None:
        mask = adata.obs["cell_type"].isin(CELL_TYPES_OF_INTEREST)
        print(
            f"Restricting to CELL_TYPES_OF_INTEREST: "
            f"{len(mask)} / {adata.n_obs} cells kept"
        )
        adata = adata[mask].copy()

    # 3.05
    PRC2_GENESETS = sequentially_reduce_genesets(PRC2_GENESETS_ORIGINAL, reduction_order)

    # 3.1 Compute PRC2/H3K27 module scores per cell
    sig_names = list(PRC2_GENESETS.keys())
    if len(sig_names) == 0:
        raise ValueError("PRC2_GENESETS is empty. Please define at least one gene set.")
    
    used_sig_names = []
    
    for sig_name, gene_list in PRC2_GENESETS.items():
        print(f"Computing module score for signature: {sig_name}")
        scores, present, missing = compute_module_score(adata, gene_list, use_raw_preferred=True)
    
        if len(present) == 0:
            print(f"  {sig_name}: 0 genes found, skipping this signature.")
            continue
    
        print(
            f"  {sig_name}: {len(present)} genes found, {len(missing)} not found "
            f"(missing: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''})"
        )
        adata.obs[f"module_{sig_name}"] = scores
        used_sig_names.append(sig_name)
    
    if not used_sig_names:
        raise ValueError("No valid signatures had any genes in this AnnData object.")
    
    # 3.2 Build a tidy DataFrame with obs + module scores
    obs_cols = ["Sample_ID", "Cancer_Type", "Response_Status", "cell_type"]
    module_cols = [f"module_{s}" for s in used_sig_names]

    df_cells = adata.obs[obs_cols + module_cols].copy()

    # 3.3 Aggregate to patient x cell_type level (mean of modules)
    print("Aggregating module scores by (Sample_ID, cell_type)...")
    grouped = (
        df_cells
        .groupby(["Sample_ID", "cell_type"], observed=True)[module_cols]
        .mean()
        .reset_index()
    )

    # 3.4 Pivot to wide format: one row per Sample_ID
    print("Pivoting to wide patient-level table...")
    wide = grouped.pivot_table(
        index="Sample_ID",
        columns="cell_type",
        values=module_cols,
        aggfunc="mean",
    )

    # wide has MultiIndex columns (module, cell_type), so flatten them
    wide_cols = []
    for (module_name, cell_type) in wide.columns:
        # module_name looks like "module_PRC2_core"
        # final col name: "<celltype>__<module_name>"
        wide_cols.append(f"{cell_type}__{module_name}")
    wide.columns = wide_cols

    wide = wide.reset_index()  # bring Sample_ID back as a column

    # 3.5 Extract per-patient metadata (Cancer_Type, Response_Status)
    # Drop duplicates in case multiple cells per sample
    meta = (
        adata.obs[["Sample_ID", "Cancer_Type", "Response_Status"]]
        .drop_duplicates(subset=["Sample_ID"])
        .copy()
    )

    # Map to coarse Aggressive_Label
    meta["Aggressive_Label"] = meta["Response_Status"].map(map_response_status)

    # Merge
    print("Merging metadata with wide feature table...")
    final_df = meta.merge(wide, on="Sample_ID", how="left")

    # 3.6 Save to CSV
    print(f"Saving patient-level table to: {out_csv_path}")
    final_df.to_csv(out_csv_path, index=False)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python build_prc2_patient_table.py <input.h5ad> <output.csv>\n\n"
            "Example:\n"
            "  python build_prc2_patient_table.py "
            "/mnt/f/H3K27/Data/scRNA/scRNA_final.h5ad "
            "/mnt/f/H3K27/Data/PRC2_patient_table.csv"
        )
        sys.exit(1)

    h5ad_path = sys.argv[1]
    out_csv_path = sys.argv[2]
    main(h5ad_path, out_csv_path)
