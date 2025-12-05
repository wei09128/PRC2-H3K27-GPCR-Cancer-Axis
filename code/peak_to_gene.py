import anndata as ad
import pandas as pd
import numpy as np
import pyranges as pr
from scipy import sparse
import re
import time
import os
import argparse # New import for command-line arguments

def parse_peaks_from_var(adata):
    """
    Parses peak coordinates from the AnnData's .var DataFrame index or columns.
    Returns a DataFrame containing 'Chrom', 'Start', 'End', and 'peak_idx'.
    """
    var = adata.var.copy()
    var["peak_index_col"] = var.index.astype(str)
    var = var.reset_index(drop=True)

    if {"chrom", "start", "end"}.issubset(var.columns):
        print("✓ Found separate chrom/start/end columns.")
        df = var.loc[:, ["chrom", "start", "end"]].copy()
        df.columns = ["Chrom", "Start", "End"]
        df["Start"] = df["Start"].astype(int)
        df["End"] = df["End"].astype(int)
    else:
        peak_col = None
        if "peak" in var.columns:
            peak_col = var["peak"]
            print("✓ Found explicit 'peak' column.")
        elif var["peak_index_col"].str.contains(r":\d+-\d+$").any():
            peak_col = var["peak_index_col"]
            print("✓ Using index content as peak coordinates.")
        else:
            raise ValueError("No valid peak coordinates found in .var.")

        # Regular expression to extract coordinates (e.g., 'chr1:100-200')
        parts = peak_col.str.extract(r"^([^:]+):(\d+)-(\d+)$")
        invalid = parts.isnull().any(axis=1)
        if invalid.any():
            bad = peak_col[invalid].head()
            raise ValueError(f"Malformed peak entries (e.g., {bad.tolist()})")

        df = pd.DataFrame({
            "Chrom": parts[0].astype(str),
            "Start": parts[1].astype(int),
            "End": parts[2].astype(int)
        })

    df["peak_idx"] = np.arange(df.shape[0])
    return df


def load_genes_from_gtf(gtf_path, feature_type="gene", name_attr_candidates=("gene_name", "gene_id")):
    """
    Loads gene features from a GTF file and returns a clean DataFrame.
    Adjusts Start to be 0-based (Start - 1).
    """
    print(f"Loading genes from GTF: {gtf_path}")
    cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]
    
    try:
        gtf = pd.read_csv(gtf_path, sep="\t", comment="#", names=cols, dtype={"chrom": str})
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: GTF file not found at {gtf_path}")

    genes = gtf[gtf["feature"] == feature_type].copy()

    def parse_name(attr):
        for n in name_attr_candidates:
            m = re.search(fr'{n} "([^"]+)"', attr)
            if m:
                return m.group(1)
        return attr

    genes["GeneName"] = genes["attributes"].apply(parse_name)
    df = genes[["chrom", "start", "end", "strand", "GeneName"]].rename(
        columns={"chrom": "Chrom", "start": "Start", "end": "End"}
    )
    # Convert to 0-based start coordinate for consistency (GTF is 1-based)
    df["Start"] = df["Start"].astype(int) - 1 
    df["End"] = df["End"].astype(int)
    df.reset_index(drop=True, inplace=True)
    print(f"✓ Loaded {len(df)} genes of type '{feature_type}'.")
    return df


def peaks_to_gene_matrix(adata, genes_df, promoter_upstream=2000, promoter_downstream=500, use_promoter_and_body=True):
    """
    Calculates gene accessibility by aggregating peak counts within gene bodies and promoters.
    """
    t0 = time.time()
    peaks_df = parse_peaks_from_var(adata)
    print(f"Parsed {len(peaks_df)} peaks.")

    # --- 1. Define Gene Regions ---
    # Build promoter ranges
    promoters = pd.DataFrame({
        "Chromosome": genes_df["Chrom"],
        "Start": (genes_df["Start"] - promoter_upstream).clip(lower=0),
        "End": genes_df["Start"] + promoter_downstream,
        "GeneName": genes_df["GeneName"]
    })
    intervals = [promoters]

    if use_promoter_and_body:
        # Build gene body ranges
        bodies = pd.DataFrame({
            "Chromosome": genes_df["Chrom"],
            "Start": genes_df["Start"],
            "End": genes_df["End"],
            "GeneName": genes_df["GeneName"]
        })
        intervals.append(bodies)

    genes_pr = pr.PyRanges(pd.concat(intervals, ignore_index=True))
    pr_peaks = pr.PyRanges(peaks_df.rename(columns={"Chrom": "Chromosome"}))

    # --- 2. Intersect Peaks and Genes ---
    print(f"⏳ Intersecting peaks ↔ gene regions (Promoter: +{promoter_downstream}/-{promoter_upstream}, Body: {use_promoter_and_body}) ...")
    ov = pr_peaks.join(genes_pr)
    ov_df = ov.df[["Start", "End", "GeneName"]].copy()

    if ov_df.empty:
        raise ValueError("No overlaps found between peaks and gene regions.")

    # --- 3. Build Peak-to-Gene (P2G) Mapping Matrix ---
    # Map peak indices directly from position
    merge_df = peaks_df.merge(ov_df, on=["Start", "End"], how="inner")[["peak_idx", "GeneName"]]
    merge_df.drop_duplicates(inplace=True)

    genes_unique = merge_df["GeneName"].unique()
    gene2idx = {g: i for i, g in enumerate(genes_unique)}
    row = merge_df["peak_idx"].to_numpy(dtype=int)
    col = merge_df["GeneName"].map(gene2idx).to_numpy(dtype=int)
    data = np.ones(len(row), dtype=np.int8)

    # P2G is a sparse matrix: Peaks (rows) x Genes (columns)
    P2G = sparse.csr_matrix((data, (row, col)), shape=(adata.shape[1], len(genes_unique)))
    print(f"✓ Built sparse peaks→genes map: {P2G.shape}, {P2G.nnz} overlaps.")

    # --- 4. Aggregate Cell Counts to Genes ---
    # Choose which matrix to use as "counts"
    if "raw_counts" in adata.layers:
        print("✓ Using adata.layers['raw_counts'] as input (peak counts).")
        X = adata.layers["raw_counts"]
    elif "counts" in adata.layers:
        print("✓ Using adata.layers['counts'] as input (peak counts).")
        X = adata.layers["counts"]
    else:
        # Fallback: use X, but warn loudly because this is usually transformed
        print("⚠️ No 'raw_counts' or 'counts' layer found. Using adata.X as input.")
        X = adata.X

    X_csr = X if sparse.issparse(X) else sparse.csr_matrix(X)

    # X_genes = X_peaks . P2G (Cells x Peaks) * (Peaks x Genes) = (Cells x Genes)
    gene_mat = X_csr.dot(P2G).tocsr()

    print(f"✓ Completed peak-to-gene aggregation in {time.time() - t0:.1f}s")
    return gene_mat, genes_unique.tolist()


    print(f"✓ Completed peak-to-gene aggregation in {time.time() - t0:.1f}s")
    return gene_mat, genes_unique.tolist()


def main():
    """Main execution function to parse arguments and run the aggregation."""
    parser = argparse.ArgumentParser(
        description="Convert scATAC/scChIP peak matrix (AnnData) to a gene accessibility matrix.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--input_h5ad',
        type=str,
        required=True,
        help="Path to the input AnnData file containing the peak matrix (e.g., /mnt/f/H3K27/Data/scChi.h5ad)"
    )
    parser.add_argument(
        '--output_h5ad',
        type=str,
        required=True,
        help="Path to save the resulting gene accessibility AnnData file (e.g., /mnt/f/H3K27/Data/scChi/scChi_accessibility.h5ad)"
    )
    parser.add_argument(
        '--gtf_path',
        type=str,
        default="/mnt/f/H3K27/gencode.v47.annotation.gtf", # Set a sensible default if possible
        help="Path to the genome annotation file (GTF format)."
    )
    
    args = parser.parse_args()
    
    input_path = args.input_h5ad
    out_path = args.output_h5ad
    gtf_path = args.gtf_path

    print("=====================================================")
    print(f"Input AnnData: {input_path}")
    print(f"Output AnnData: {out_path}")
    print(f"GTF Annotation: {gtf_path}")
    print("=====================================================")

    # --- 1. Load Data ---
    try:
        adata = ad.read_h5ad(input_path)
    except FileNotFoundError:
        print(f"Error: AnnData file not found at {input_path}. Exiting.")
        return
    
    genes_df = load_genes_from_gtf(gtf_path)
    
    # --- 2. Calculate Gene Matrix ---
    gene_mat, genes = peaks_to_gene_matrix(adata, genes_df)
    
    # --- 3. Create and Save New AnnData Object ---
    adata_genes = ad.AnnData(
        X=gene_mat, 
        obs=adata.obs.copy(), 
        var=pd.DataFrame(index=genes)
    )
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save the result
    adata_genes.write_h5ad(out_path)
    print("✓ Saved gene accessibility matrix successfully!")


if __name__ == "__main__":
    main()
