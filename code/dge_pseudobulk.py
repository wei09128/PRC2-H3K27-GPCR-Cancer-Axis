#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import os
import argparse
import sys
from typing import List, Dict, Union, Any
from scipy.sparse import issparse

# =========================
# Configuration Constants
# =========================
MIN_EXPRESSION_PCT = 0.10   # 10% min expression in at least one group
FDR_CUTOFF = 0.05           # FDR threshold
LFC_CUTOFF = 0.50           # |logFC| threshold
TARGET_GENE_COUNT = 0       # if >0 and no significant genes, take top-N (up+down)
MIN_CELLS_PER_GROUP = 100   # min cells per group (cell-level Wilcoxon)

# --- Pseudobulk Config ---
PSEUDOBULK_MIN_CELLS_PER_SAMPLE = 7   # min cells per (patient x condition x cell_type)
MIN_PATIENTS_PER_GROUP = 3             # per condition (Aggressive vs Control)

# --- USER-DEFINED GROUPING ---
AGGRESSIVE_STATUS_GROUPS = ['Persister', 'Recurrent', 'Residual', 'Resistant','Aggressive']
CONTROL_STATUS_GROUPS = ['less-Aggressive']
NEW_STATUS_KEY = 'aggressiveness_group'


# =========================
# Helpers (Pseudobulk)
# =========================
def _mode(series: pd.Series):
    return series.mode().iat[0] if not series.mode().empty else None


def aggregate_pseudobulk_by_celltype(
    adata: sc.AnnData,
    cell_type_key: str,
    status_key: str,
    patient_key: str,
    raw_layer: str,
    batch_key: Union[str, None] = None,
    min_cells_per_sample: int = PSEUDOBULK_MIN_CELLS_PER_SAMPLE
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build pseudobulk matrices per cell type: sum raw UMIs by (patient_id, condition).
    Returns: { cell_type: {'counts': genes x samples, 'coldata': samples x meta} }
    """
    if raw_layer not in adata.layers:
        raise ValueError(f"Raw count layer '{raw_layer}' not in adata.layers")

    out = {}
    for ct in adata.obs[cell_type_key].unique():
        sub = adata[adata.obs[cell_type_key] == ct].copy()
        if sub.n_obs == 0:
            continue

        cols = [patient_key, status_key, cell_type_key]
        if batch_key and batch_key in sub.obs.columns:
            cols.append(batch_key)
        obs = sub.obs[cols].copy()
        obs["__grp__"] = list(zip(obs[patient_key], obs[status_key]))

        # enforce min cells per (patient, condition)
        grp_sizes = obs["__grp__"].value_counts()
        keep_groups = set([g for g, n in grp_sizes.items() if n >= min_cells_per_sample])
        if not keep_groups:
            continue

        keep_mask = obs["__grp__"].apply(lambda g: g in keep_groups).values
        sub = sub[keep_mask].copy()
        if sub.n_obs == 0:
            continue

        cols = [patient_key, status_key, cell_type_key]
        if batch_key and batch_key in sub.obs.columns:
            cols.append(batch_key)
        obs = sub.obs[cols].copy()
        obs["__grp__"] = list(zip(obs[patient_key], obs[status_key]))

        groups = obs.groupby("__grp__").indices  # dict: (patient, condition) -> row idx
        genes = sub.var_names
        sample_cols = []
        sample_names = []

        for (pat, cond), idx in groups.items():
            Xg = sub.layers[raw_layer][list(idx), :]
            if issparse(Xg):
                col = np.asarray(Xg.sum(axis=0)).ravel()
            else:
                col = Xg.sum(axis=0)
            sample_cols.append(col)
            sample_names.append(f"{pat}__{cond}")

        if not sample_cols:
            continue

        counts = pd.DataFrame(np.vstack(sample_cols).T, index=genes, columns=sample_names)

        # coldata
        meta_rows = []
        for (pat, cond), idx in groups.items():
            row = {"sample": f"{pat}__{cond}",
                   "patient_id": pat,
                   "condition": cond,
                   "cell_type": ct}
            if batch_key and batch_key in sub.obs.columns:
                row["batch"] = _mode(sub.obs.iloc[list(idx)][batch_key])
            meta_rows.append(row)
        coldata = pd.DataFrame(meta_rows).set_index("sample").loc[counts.columns]

        # require both groups and min patients per group
        n_by_cond = coldata.groupby("condition")["patient_id"].nunique()
        if not {"Aggressive", "Control"}.issubset(set(n_by_cond.index)):
            continue
        if (n_by_cond.get("Aggressive", 0) < MIN_PATIENTS_PER_GROUP) or \
           (n_by_cond.get("Control", 0) < MIN_PATIENTS_PER_GROUP):
            continue

        out[ct] = {"counts": counts, "coldata": coldata}

    return out


def run_edger_pseudobulk(
    counts: pd.DataFrame,
    coldata: pd.DataFrame,
    include_batch: bool = True,
    output_prefix: str = "pseudobulk",
    coef_name: str = "conditionAggressive"
) -> pd.DataFrame:
    """
    Run edgeR (QLF) via rpy2. Returns DataFrame with logFC, FDR, etc.
    If rpy2/edgeR unavailable, save CSVs and raise ImportError with R snippet.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        r = ro.r
        r("suppressMessages(library(edgeR))")
        r("suppressMessages(library(limma))")
    except Exception as e:
        counts.to_csv(f"{output_prefix}_counts.csv")
        coldata.to_csv(f"{output_prefix}_coldata.csv")
        raise ImportError(
            f"rpy2/edgeR not available. Saved:\n  {output_prefix}_counts.csv\n  {output_prefix}_coldata.csv\n"
            f"Run in R:\n"
            f"counts <- read.csv('{output_prefix}_counts.csv', row.names=1)\n"
            f"coldata <- read.csv('{output_prefix}_coldata.csv', row.names=1)\n"
            f"coldata$condition <- factor(coldata$condition, levels=c('Control','Aggressive'))\n"
            f"design <- model.matrix(~ condition{' + batch' if include_batch and 'batch' in coldata.columns else ''}, data=coldata)\n"
            f"y <- DGEList(counts)\nkeep <- filterByExpr(y, design)\ny <- y[keep,,keep.lib.sizes=FALSE]\n"
            f"y <- calcNormFactors(y)\ny <- estimateDisp(y, design)\nfit <- glmQLFit(y, design)\n"
            f"tab <- topTags(glmQLFTest(fit, coef='conditionAggressive'), n=Inf)$table\n"
        ) from e

    r = ro.r
    from rpy2.robjects import pandas2ri
    r_counts = pandas2ri.py2rpy(counts.astype(int))
    r_coldata = pandas2ri.py2rpy(coldata.copy())
    r.assign("counts", r_counts)
    r.assign("coldata", r_coldata)

    design_formula = "~ condition"
    if include_batch and "batch" in coldata.columns:
        design_formula += " + batch"

    r(f"""
        coldata$condition <- factor(coldata$condition, levels=c('Control','Aggressive'))
        y <- DGEList(counts)
        keep <- filterByExpr(y, model.matrix({design_formula}, data=coldata))
        y <- y[keep, , keep.lib.sizes=FALSE]
        y <- calcNormFactors(y, method="TMM")
        design <- model.matrix({design_formula}, data=coldata)
        y <- estimateDisp(y, design)
        fit <- glmQLFit(y, design)
        qlf <- glmQLFTest(fit, coef="{coef_name}")
        tab <- topTags(qlf, n=Inf)$table
        tab$gene <- rownames(tab)
    """)

    res = r("tab")
    res_df = pd.DataFrame(np.array(res)).T
    res_df.columns = list(res.names)
    # ensure numeric
    for col in ["logFC", "FDR"]:
        res_df[col] = pd.to_numeric(res_df[col], errors="coerce")
    return res_df


def perform_pseudobulk_dge(
    adata: sc.AnnData,
    cancer_type: str,
    cell_type_key: str,
    status_key: str,
    patient_key: str,
    raw_layer: str,
    batch_key: Union[str, None],
    output_dir: str = '.'
) -> Dict[str, Dict[str, Any]]:
    """
    Pseudobulk DGE per cell type using edgeR. Falls back to saving CSVs if rpy2 not available.
    Returns: {cell_type: {'genes': [up+down], 'up_count': int}}
    """
    pb = aggregate_pseudobulk_by_celltype(
        adata=adata,
        cell_type_key=cell_type_key,
        status_key=status_key,
        patient_key=patient_key,
        raw_layer=raw_layer,
        batch_key=batch_key,
        min_cells_per_sample=PSEUDOBULK_MIN_CELLS_PER_SAMPLE
    )

    if not pb:
        print("Warning: No cell types qualified for pseudobulk (check thresholds and group balance).")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    top_genes_for_pathway: Dict[str, Dict[str, Any]] = {}

    for ct, dd in pb.items():
        counts = dd["counts"]
        coldata = dd["coldata"]
        prefix = os.path.join(output_dir, f"pseudobulk_{ct.replace(' ','_')}")
        print(f"Running edgeR pseudobulk for cell type: {ct} (samples={counts.shape[1]})")

        try:
            res_df = run_edger_pseudobulk(
                counts=counts,
                coldata=coldata,
                include_batch=("batch" in coldata.columns),
                output_prefix=prefix,
                coef_name="conditionAggressive"
            )
        except ImportError as e:
            print("\nNOTE:", e, "\nSkipping in-Python DGE for this cell type.")
            continue

        out_csv = f"{prefix}_edgeR_DGE.csv"
        res_df.to_csv(out_csv, index=False)
        print(f"✅ Saved pseudobulk edgeR results: {out_csv}")

        # Select genes for pathway analysis
        sig = res_df[(res_df["FDR"] < FDR_CUTOFF) & (res_df["logFC"].abs() > LFC_CUTOFF)].copy()
        sig.sort_values("logFC", ascending=False, inplace=True)
        up_genes = sig[sig["logFC"] > 0]["gene"].tolist()
        down_genes = sig[sig["logFC"] < 0]["gene"].tolist()

        if len(sig) == 0 and TARGET_GENE_COUNT > 0:
            res_df.sort_values("logFC", ascending=False, inplace=True)
            N = TARGET_GENE_COUNT // 2
            up_genes = res_df[res_df["logFC"] > 0]["gene"].head(N).tolist()
            down_genes = res_df[res_df["logFC"] < 0]["gene"].tail(N).tolist()

        gene_list = up_genes + down_genes
        up_count = len(up_genes)
        top_genes_for_pathway[ct] = {"genes": gene_list, "up_count": up_count}

        gl_csv = f"{prefix}_genes_for_pathway.csv"
        pd.DataFrame({"gene": gene_list}).to_csv(gl_csv, index=False)
        print(f"✅ Saved genes for pathway analysis: {gl_csv}")

    return top_genes_for_pathway


# =========================
# Existing Wilcoxon DGE (cell-level)
# =========================
def perform_dge_analysis(
    adata: sc.AnnData,
    cancer_type: str,
    cell_type_key: str,
    status_key: str,
    raw_layer: str,
    reference_group: str = 'Control',
    output_dir: str = '.'
) -> Dict[str, Dict[str, Any]]:
    """
    Cell-level Wilcoxon DGE with hybrid thresholding.
    """
    print(f"Starting DGE analysis using layer: '{raw_layer}'")
    print(f"Comparing groups in '{status_key}' with reference: '{reference_group}'")

    results_dfs = []
    top_genes_for_pathway: Dict[str, Dict[str, Any]] = {}
    cell_types = adata.obs[cell_type_key].unique()

    for ct_name in cell_types:
        adata_subset = adata[adata.obs[cell_type_key] == ct_name].copy()
        groups_present = adata_subset.obs[status_key].unique()

        if not all(g in groups_present for g in ['Aggressive', reference_group]):
            print(f" -> Skipping {ct_name}: Missing 'Aggressive' or '{reference_group}'.")
            continue

        group_counts = adata_subset.obs[status_key].value_counts()
        aggressive_count = group_counts.get('Aggressive', 0)
        control_count = group_counts.get(reference_group, 0)
        if aggressive_count < MIN_CELLS_PER_GROUP or control_count < MIN_CELLS_PER_GROUP:
            print(f" -> Skipping {ct_name}: counts too low. Aggressive={aggressive_count}, Control={control_count} (Min {MIN_CELLS_PER_GROUP}).")
            continue

        print(f"Running Wilcoxon test for cell type: {ct_name}")
        sc.tl.rank_genes_groups(
            adata_subset,
            groupby=status_key,
            groups=['Aggressive'],
            reference=reference_group,
            method='wilcoxon',
            layer=raw_layer,
            key_added='wilcoxon_dge'
        )

        results_df = sc.get.rank_genes_groups_df(adata_subset, group='Aggressive', key='wilcoxon_dge')

        group_filter = adata_subset.obs[status_key] == 'Aggressive'
        ref_filter = adata_subset.obs[status_key] == reference_group
        raw_counts = adata_subset.layers[raw_layer]

        agg_subset = raw_counts[group_filter.values, :]
        ref_subset = raw_counts[ref_filter.values, :]

        def _pct_expr(mat):
            if issparse(mat):
                return np.mean(mat.toarray() > 0, axis=0)
            arr = np.asarray(mat)
            return np.mean(arr > 0, axis=0)

        results_df['pct_agg'] = _pct_expr(agg_subset)
        results_df['pct_ctrl'] = _pct_expr(ref_subset)
        results_df['cell_type'] = ct_name

        exp_filter = (results_df['pct_agg'] >= MIN_EXPRESSION_PCT) | (results_df['pct_ctrl'] >= MIN_EXPRESSION_PCT)
        filtered_df = results_df[exp_filter].copy()

        # Significant genes: FDR < 0.05 & |LFC| > threshold
        statistically_significant_df = filtered_df[
            (filtered_df['pvals_adj'] < FDR_CUTOFF) &
            (abs(filtered_df['logfoldchanges']) > LFC_CUTOFF)
        ].copy()
        statistically_significant_df.sort_values(by='logfoldchanges', ascending=False, inplace=True)

        if len(statistically_significant_df) >= TARGET_GENE_COUNT:
            up_genes = list(statistically_significant_df[statistically_significant_df['logfoldchanges'] > 0]['names'])
            down_genes = list(statistically_significant_df[statistically_significant_df['logfoldchanges'] < 0]['names'])
            print(f" -> {ct_name}: {len(up_genes) + len(down_genes)} DEGs (FDR<{FDR_CUTOFF}, |LFC|>{LFC_CUTOFF}).")
        else:
            filtered_df.sort_values(by='logfoldchanges', ascending=False, inplace=True)
            N = TARGET_GENE_COUNT // 2
            up_regulated = filtered_df[filtered_df['logfoldchanges'] > 0].head(N)
            down_regulated = filtered_df[filtered_df['logfoldchanges'] < 0].tail(N)
            up_genes = list(up_regulated['names'])
            down_genes = list(down_regulated['names'])
            print(f" -> {ct_name}: Only {len(statistically_significant_df)} DEGs; fallback Top {len(up_genes)} Up/{len(down_genes)} Down.")

        top_genes_list = up_genes + down_genes
        up_count = len(up_genes)
        top_genes_for_pathway[ct_name] = {'genes': top_genes_list, 'up_count': up_count}
        results_dfs.append(filtered_df)

    if not results_dfs:
        print("\nWarning: No cell types had both groups for comparison.")
        return {}

    full_results_path = os.path.join(output_dir, 'dge_wilcoxon_full_results.csv')
    pd.concat(results_dfs).to_csv(full_results_path, index=False)
    print(f"\n✅ Full DGE results saved to: {full_results_path}")

    top_genes_path = os.path.join(output_dir, 'dge_genes_for_pathway_analysis.csv')
    gene_lists_only = {k: v['genes'] for k, v in top_genes_for_pathway.items()}
    if gene_lists_only:
        pd.DataFrame({k: pd.Series(v) for k, v in gene_lists_only.items()}).fillna('').to_csv(top_genes_path, index=False)
        print(f"✅ Genes for pathway analysis saved to: {top_genes_path}")

    return top_genes_for_pathway


# =========================
# Main
# =========================
def main():
    # 👇 must be the first executable line in main()
    global PSEUDOBULK_MIN_CELLS_PER_SAMPLE, MIN_PATIENTS_PER_GROUP

    parser = argparse.ArgumentParser(
        description="DGE analysis (Wilcoxon cell-level or pseudobulk per patient) for aggressiveness.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # File/Path
    parser.add_argument('--h5ad_path', type=str, required=True, help="Path to AnnData .h5ad file.")
    parser.add_argument('--cancer_type', type=str, default=None, help="Optional cancer type filter, e.g., 'BC'.")
    parser.add_argument('--output_dir', type=str, default='./dge_results', help="Output directory.")

    # Observation keys
    parser.add_argument('--cell_type_key', type=str, default='cell_type', help="obs column for cell type.")
    parser.add_argument('--status_key', type=str, default='Response_Status', help="obs column for original status.")
    parser.add_argument('--raw_layer', type=str, default='raw_counts', help="Layer with raw UMI counts.")

    # Pseudobulk options (define ONCE)
    parser.add_argument('--run_pseudobulk', action='store_true', help="Run pseudobulk DGE instead of Wilcoxon.")
    parser.add_argument('--patient_key', type=str, default='patient_id', help="obs column with patient/donor ID.")
    parser.add_argument('--batch_key', type=str, default=None, help="Optional obs column for technical batch.")
    parser.add_argument('--pseudobulk_min_cells', type=int, default=PSEUDOBULK_MIN_CELLS_PER_SAMPLE,
                        help="Min cells per (patient x condition x cell_type).")
    parser.add_argument('--min_patients_per_group', type=int, default=MIN_PATIENTS_PER_GROUP,
                        help="Min patients per group (Aggressive/Control) in pseudobulk.")

    args = parser.parse_args()


    PSEUDOBULK_MIN_CELLS_PER_SAMPLE = args.pseudobulk_min_cells
    MIN_PATIENTS_PER_GROUP = args.min_patients_per_group

    # Load data
    try:
        adata = sc.read_h5ad(args.h5ad_path)
        adata.obs.columns = adata.obs.columns.str.strip()
        for col in {args.patient_key, args.batch_key, args.status_key, args.cell_type_key, 'Cancer_Type'} - {None}:
            if col in adata.obs.columns:
                adata.obs[col] = adata.obs[col].astype(str).str.strip()

    except FileNotFoundError:
        print(f"Error: AnnData file not found at {args.h5ad_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes.")

    if args.raw_layer not in adata.layers:
        print(f"Error: Raw count layer '{args.raw_layer}' not found in adata.layers.")
        sys.exit(1)

    # Optional cancer filter
    if args.cancer_type is not None:
        if 'Cancer_Type' in adata.obs.columns:
            print(f"\nFiltering to Cancer_Type == '{args.cancer_type}'...")
            original_n_obs = adata.n_obs
            adata = adata[adata.obs['Cancer_Type'] == args.cancer_type].copy()
            print(f"Filtered from {original_n_obs} to {adata.n_obs} cells.")
            if adata.n_obs == 0:
                print(f"Error: No cells with Cancer_Type == '{args.cancer_type}'")
                sys.exit(1)
        else:
            print(f"Warning: 'Cancer_Type' not found in obs; skipping filter.")
    else:
        print("\nNo --cancer_type specified. Using all cancer types.")

    # Group statuses into Aggressive vs Control
    print(f"\nPreprocessing: Grouping '{args.status_key}' into Aggressive vs Control...")
    print(f"  Aggressive: {AGGRESSIVE_STATUS_GROUPS}")
    print(f"  Control: {CONTROL_STATUS_GROUPS}")

    adata.obs[NEW_STATUS_KEY] = 'Undefined'
    control_mask = adata.obs[args.status_key].isin(CONTROL_STATUS_GROUPS)
    adata.obs.loc[control_mask, NEW_STATUS_KEY] = 'Control'
    agg_mask = adata.obs[args.status_key].isin(AGGRESSIVE_STATUS_GROUPS)
    adata.obs.loc[agg_mask, NEW_STATUS_KEY] = 'Aggressive'

    original_n_obs = adata.n_obs
    adata = adata[adata.obs[NEW_STATUS_KEY] != 'Undefined'].copy()
    if adata.n_obs < original_n_obs:
        print(f"Warning: {original_n_obs - adata.n_obs} cells excluded (not Aggressive/Control).")

    if not all(g in adata.obs[NEW_STATUS_KEY].unique() for g in ['Aggressive', 'Control']):
        print("Error: One or both groups missing after grouping. Check mapping.")
        sys.exit(1)

    print(f"\nNew groups in '{NEW_STATUS_KEY}':")
    print(adata.obs[NEW_STATUS_KEY].value_counts())

    print("\n--- Mapping of Original Status to New Groups ---")
    status_mapping = adata.obs.groupby([args.status_key, NEW_STATUS_KEY]).size().unstack(fill_value=0)
    try:
        print(status_mapping.to_markdown(numalign="left", stralign="left"))
    except Exception:
        print(status_mapping)
    print("------------------------------------------------------------------")

    # Cell counts per cell type
    print("\n--- Cell Counts per Cell Type (Aggressive vs Control) ---")
    cell_type_counts = adata.obs.groupby([args.cell_type_key, NEW_STATUS_KEY]).size().unstack(fill_value=0)
    if 'Aggressive' not in cell_type_counts.columns:
        cell_type_counts['Aggressive'] = 0
    if 'Control' not in cell_type_counts.columns:
        cell_type_counts['Control'] = 0
    cell_type_counts['Total'] = cell_type_counts['Aggressive'] + cell_type_counts['Control']
    cell_type_counts = cell_type_counts[['Aggressive', 'Control', 'Total']]
    try:
        print(cell_type_counts.to_markdown(numalign="left", stralign="left"))
    except Exception:
        print(cell_type_counts)
    print("------------------------------------------------------------------")

    # Run analysis
    if args.run_pseudobulk:
        print("\n=== Running PSEUDOBULK DGE per patient ===")
        top_genes = perform_pseudobulk_dge(
            adata=adata,
            cancer_type=args.cancer_type,
            cell_type_key=args.cell_type_key,
            status_key=NEW_STATUS_KEY,
            patient_key=args.patient_key,
            raw_layer=args.raw_layer,
            batch_key=args.batch_key,
            output_dir=args.output_dir
        )
    else:
        print("\n=== Running cell-level Wilcoxon DGE ===")
        top_genes = perform_dge_analysis(
            adata=adata,
            cancer_type=args.cancer_type,
            cell_type_key=args.cell_type_key,
            status_key=NEW_STATUS_KEY,
            raw_layer=args.raw_layer,
            reference_group='Control',
            output_dir=args.output_dir
        )

    # Print selections
    print("\n\n--- Genes Selected for Pathway Analysis ---")
    if not top_genes:
        print("No gene lists were produced.")
    else:
        for ct, data in top_genes.items():
            genes = data['genes']
            up_count = data['up_count']
            up_genes = genes[:up_count]
            down_genes = genes[up_count:]
            print(f"Cell Type: {ct} ({len(genes)} genes)")
            if up_genes:
                print(f"  Up ({len(up_genes)}): {', '.join(up_genes[:25])}{' ...' if len(up_genes)>25 else ''}")
            if down_genes:
                print(f"  Down ({len(down_genes)}): {', '.join(down_genes[:25])}{' ...' if len(down_genes)>25 else ''}")
            print("-" * 20)


if __name__ == "__main__":
    if 'ipykernel' in sys.modules or 'jupyter' in sys.modules:
        print("Note: Running interactively. Use CLI for full script execution.")
    else:
        main()
