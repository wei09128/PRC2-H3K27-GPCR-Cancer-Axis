#!/usr/bin/env python3
"""
tf_activity_pipeline.py - FIXED VERSION

End-to-end pipeline with fixes for:
  - FIMO file parsing issues (warning lines, missing column names)
  - Coordinate matching (FASTA: chr:start-end, FIMO: genomic coords)
  - Memory-efficient processing
"""

import os, glob, sys
import numpy as np
import pandas as pd
from scipy import sparse
import h5py
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ==================================================
# CONFIG
# ==================================================

H5AD_PATH   = "/mnt/f/H3K27/Data/scATAC/scATAC_peaks_QC_BC_float64_clean.h5ad"
FASTA_FILE  = "/mnt/f/H3K27/Data/TF/scATAC_peak_sequences.fasta"

FIMO_DIR    = "/mnt/f/H3K27/Data/TF/fimo_parallel"
FIMO_GLOB   = os.path.join(FIMO_DIR, "fimo_*.txt")
COMBINED_FIMO_PATH = os.path.join(FIMO_DIR, "fimo_partial_combined.txt")
DEDUP_FIMO_PATH    = os.path.join(FIMO_DIR, "fimo_dedup.txt")

OUTPUT_DIR  = "/mnt/f/H3K27/Data/TF/analysis"
OUTPUT_H5AD_WITH_TF = "/mnt/f/H3K27/Data/scATAC/scATAC_peaks_QC_BC_float64_clean_with_TF.h5ad"

FIMO_SCORE_THRESH = 10.0
MIN_CELLS_PER_GROUP = 10
DO_DEDUP = False   # Already have deduped file

# ==================================================
# STEP 0b: DEDUP FIMO (FIXED - handles missing col names)
# ==================================================

def dedup_fimo_hits(in_path=COMBINED_FIMO_PATH, out_path=DEDUP_FIMO_PATH, chunksize=2_000_000):
    """
    Chunked dedup with FIXED column name handling
    """
    print("\n" + "="*60)
    print("Deduplicating FIMO hits (chunked)")
    print("="*60)

    best_rows = {}

    # Define column names explicitly
    column_names = ['motif_id', 'motif_alt_id', 'sequence_name', 'start', 
                   'stop', 'strand', 'score', 'p-value', 'q-value', 'matched_sequence']

    total_in = 0
    chunk_i = 0
    
    try:
        # Read in chunks, skip warning line, use explicit column names
        for chunk in pd.read_csv(in_path, sep="\t", chunksize=chunksize,
                                 skiprows=1, names=column_names,
                                 on_bad_lines="skip", low_memory=False):
            chunk_i += 1
            
            # Convert numeric columns
            for c in ["start", "stop", "score", "p-value"]:
                if c in chunk.columns:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            
            chunk = chunk.dropna(subset=["start", "stop", "score"])
            total_in += len(chunk)
            print(f"  chunk {chunk_i}: {len(chunk):,} rows (running total {total_in:,})")

            for row in chunk.itertuples(index=False):
                key = (row.motif_alt_id, row.sequence_name, int(row.start), int(row.stop))
                scv = float(row.score)
                if key not in best_rows or scv > best_rows[key][0]:
                    best_rows[key] = (scv, row._asdict())

    except Exception as e:
        print(f"⚠️ Error during dedup: {e}")
        if not best_rows:
            raise

    dedup_df = pd.DataFrame([v[1] for v in best_rows.values()])
    dedup_df.to_csv(out_path, sep="\t", index=False)
    print(f"\n✓ Deduped rows: {len(dedup_df):,}")
    print(f"✓ Wrote: {out_path}")
    return out_path

# ==================================================
# STEP 1: CREATE PEAK MAPPING
# ==================================================

def create_peak_mapping(fasta_file):
    """Create coordinate → peak_index mapping with parsed coordinates"""
    print("\nCreating peak mapping from FASTA...")
    coord_to_idx = {}
    peak_ranges = {}

    with open(fasta_file, "r") as f:
        peak_idx = 0
        for line in f:
            if line.startswith(">"):
                coord = line.strip()[1:]
                coord_to_idx[coord] = peak_idx

                # Parse chr:start-end
                chrom, pos = coord.split(":")
                start, end = map(int, pos.split("-"))

                peak_ranges.setdefault(chrom, []).append((start, end, peak_idx))
                peak_idx += 1

    # Sort ranges for efficient lookup
    for chrom in peak_ranges:
        peak_ranges[chrom].sort()

    print(f"  ✓ Mapped {len(coord_to_idx):,} peaks")
    print(f"  ✓ Chromosomes: {len(peak_ranges)}")
    
    # Show chromosome stats
    chrom_counts = {ch: len(ranges) for ch, ranges in peak_ranges.items()}
    print(f"  ✓ Sample counts: chr1={chrom_counts.get('chr1', 0):,}, chr10={chrom_counts.get('chr10', 0):,}")
    
    return coord_to_idx, peak_ranges

# ==================================================
# STEP 2: LOAD scATAC + FIMO (FIXED)
# ==================================================

def fix_response_status(obs):
    """Merge Resistant → Aggressive and create Response_state"""
    if "Response_Status" in obs.columns:
        print("\nOriginal Response_Status:")
        print(obs["Response_Status"].value_counts())

        obs["Response_Status"] = obs["Response_Status"].replace("Resistant", "Aggressive")
        obs["Response_state"] = obs["Response_Status"]

        print("\nAfter merging Resistant→Aggressive:")
        print(obs["Response_state"].value_counts())
    return obs

def load_h5ad_manual(filepath):
    """Load h5ad with h5py to stay memory-light"""
    print(f"\nLoading h5ad (manual): {filepath}")
    with h5py.File(filepath, "r") as f:
        if isinstance(f["X"], h5py.Dataset):
            X = f["X"][:]
        else:
            X = sparse.csr_matrix(
                (f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]),
                shape=tuple(f["X"].attrs["shape"])
            )

        obs = pd.DataFrame()
        if "obs" in f:
            for key in list(f["obs"].keys()):
                if key == "_index": 
                    continue
                item = f["obs"][key]
                if isinstance(item, h5py.Group):
                    continue
                try:
                    data = item[:]
                    if data.dtype.kind in ["S", "O"]:
                        data = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in data])
                    obs[key] = data
                except:
                    pass

            if "_index" in f["obs"]:
                idx_data = f["obs"]["_index"][:]
                obs.index = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in idx_data]

        var = pd.DataFrame(index=list(range(X.shape[1])))

    print(f"  Cells: {X.shape[0]:,}")
    print(f"  Peaks: {X.shape[1]:,}")
    return X, obs, var

def load_fimo_results(filepath):
    """FIXED: Load FIMO with proper column handling"""
    print(f"\nLoading FIMO: {filepath}")
    
    # Check if it's the deduped file (has proper header) or combined (needs column names)
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    if first_line.startswith("Warning"):
        # Combined file - skip warning, use explicit column names
        print("  Detected combined file format (with warning)")
        column_names = ['motif_id', 'motif_alt_id', 'sequence_name', 'start', 
                       'stop', 'strand', 'score', 'p-value', 'q-value', 'matched_sequence']
        fimo = pd.read_csv(filepath, sep="\t", skiprows=1, names=column_names,
                          on_bad_lines="skip", low_memory=False)
    else:
        # Deduped file or standard FIMO output
        print("  Detected standard/deduped file format")
        fimo = pd.read_csv(filepath, sep="\t", on_bad_lines="skip", low_memory=False)
        
        # Fix unnamed columns (_7, _8) → (p-value, q-value)
        if '_7' in fimo.columns:
            fimo = fimo.rename(columns={'_7': 'p-value', '_8': 'q-value'})
            print("  ✓ Fixed column names: _7→p-value, _8→q-value")

    fimo.columns = fimo.columns.str.strip()
    
    # Convert numeric columns
    for c in ["start", "stop", "score", "p-value"]:
        if c in fimo.columns:
            fimo[c] = pd.to_numeric(fimo[c], errors="coerce")
    
    fimo = fimo.dropna(subset=["start", "stop", "score"])
    
    # Convert to int (FIMO dedup may have saved as float)
    fimo["start"] = fimo["start"].astype(int)
    fimo["stop"] = fimo["stop"].astype(int)

    print(f"  ✓ Rows: {len(fimo):,}")
    print(f"  ✓ TFs:  {fimo['motif_alt_id'].nunique():,}")
    print(f"  ✓ Columns: {list(fimo.columns)}")
    
    # Show coordinate examples
    print(f"\n  Sample coordinates:")
    for i, row in fimo.head(3).iterrows():
        print(f"    {row['sequence_name']}:{row['start']}-{row['stop']}")
    
    return fimo

# ==================================================
# STEP 3: MAP FIMO HITS TO PEAKS (FIXED for genomic coords)
# ==================================================

def map_fimo_to_peaks(fimo, peak_ranges, score_thresh=FIMO_SCORE_THRESH):
    """
    Map FIMO genomic coordinates to peak indices.
    FIMO has: chr + genomic start/stop
    FASTA has: chr:start-end
    """
    print("\n" + "="*60)
    print("Mapping FIMO hits to peaks")
    print("="*60)

    fimo_filt = fimo[fimo["score"] >= score_thresh].copy()
    print(f"After score filter (>= {score_thresh}): {len(fimo_filt):,} hits")

    if fimo_filt.empty:
        return None

    mapped_hits = []
    unmapped = 0
    progress_interval = max(1, len(fimo_filt) // 20)  # Show 20 progress updates

    for idx, row in enumerate(fimo_filt.itertuples(index=False)):
        if (idx + 1) % progress_interval == 0:
            print(f"  Progress: {idx+1:,}/{len(fimo_filt):,} ({100*(idx+1)/len(fimo_filt):.1f}%)")
        
        chrom = str(row.sequence_name).strip()
        motif_start = int(row.start)
        motif_stop = int(row.stop)

        if chrom not in peak_ranges:
            unmapped += 1
            continue

        # Find overlapping peak
        found = False
        for peak_start, peak_end, peak_idx in peak_ranges[chrom]:
            # Check if motif overlaps with peak
            if motif_start < peak_end and motif_stop > peak_start:
                mapped_hits.append({
                    "motif_alt_id": row.motif_alt_id,
                    "peak_idx": peak_idx,
                    "score": float(row.score),
                    "chrom": chrom
                })
                found = True
                break
            
            # Optimization: if we've passed the motif, stop searching
            if peak_start > motif_stop:
                break

        if not found:
            unmapped += 1

    print(f"\n  ✓ Mapped hits:   {len(mapped_hits):,}")
    print(f"  ✓ Unmapped hits: {unmapped:,}")
    print(f"  ✓ Mapping rate:  {100*len(mapped_hits)/(len(mapped_hits)+unmapped):.1f}%")

    if not mapped_hits:
        return None

    mapped_df = pd.DataFrame(mapped_hits).drop_duplicates(subset=["motif_alt_id", "peak_idx"])
    print(f"  ✓ Unique TF-peak pairs: {len(mapped_df):,}")
    
    # Show some mapping stats
    top_tfs = mapped_df['motif_alt_id'].value_counts().head(5)
    print(f"\n  Top 5 TFs by peaks mapped:")
    for tf, count in top_tfs.items():
        print(f"    {tf}: {count:,} peaks")
    
    return mapped_df

# ==================================================
# STEP 4: CREATE TF ACTIVITY MATRIX
# ==================================================

def create_tf_activity_matrix(X, mapped_fimo):
    print("\n" + "="*60)
    print("Creating TF Activity Matrix")
    print("="*60)

    if mapped_fimo is None or mapped_fimo.empty:
        print("❌ No mapped FIMO hits.")
        return None, None

    tfs = sorted(mapped_fimo["motif_alt_id"].unique())
    n_cells = X.shape[0]
    tf_activity = np.zeros((n_cells, len(tfs)), dtype=np.float32)

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    for i, tf in enumerate(tfs):
        tf_peaks = mapped_fimo.loc[mapped_fimo["motif_alt_id"] == tf, "peak_idx"].values
        valid_peaks = tf_peaks[tf_peaks < X.shape[1]]
        if len(valid_peaks) > 0:
            tf_activity[:, i] = np.asarray(X[:, valid_peaks].sum(axis=1)).flatten()

        if (i+1) % 25 == 0 or i == len(tfs)-1:
            print(f"  processed {i+1}/{len(tfs)} TFs")

    print(f"\n✓ TF activity shape: {tf_activity.shape}")
    
    # Show some stats
    nonzero_tfs = (tf_activity.sum(axis=0) > 0).sum()
    print(f"✓ TFs with activity: {nonzero_tfs}/{len(tfs)}")
    print(f"✓ Mean activity per cell: {tf_activity.sum(axis=1).mean():.2f}")
    
    return tf_activity, tfs

# ==================================================
# STEP 5: SAVE TF ACTIVITY NPZ
# ==================================================

def save_tf_npz(tf_activity, tfs, cell_ids, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tf_activity.npz")

    np.savez_compressed(
        out_path,
        tf_activity=tf_activity,
        tfs=np.array(tfs, dtype=object),
        cell_ids=np.array(cell_ids, dtype=object)
    )
    print(f"\n✓ Saved TF npz: {out_path}")
    return out_path

# ==================================================
# STEP 6: ADD TF TO AnnData + SAVE
# ==================================================

def add_tf_to_anndata(npz_path, h5ad_in=H5AD_PATH, h5ad_out=OUTPUT_H5AD_WITH_TF):
    print("\n" + "="*60)
    print("Adding TF activity to AnnData")
    print("="*60)

    print("Loading AnnData...")
    adata = anndata.read_h5ad(h5ad_in)

    print("Loading TF npz...")
    tf_data = np.load(npz_path, allow_pickle=True)
    tf_activity = tf_data["tf_activity"]
    tf_names = tf_data["tfs"]
    cell_ids = tf_data["cell_ids"]

    # Ensure order matches
    if not np.array_equal(adata.obs_names.values, cell_ids):
        print("⚠️ Cell order mismatch — reordering TF activity to match AnnData.")
        idx_map = pd.Series(np.arange(len(cell_ids)), index=cell_ids)
        order = idx_map.loc[adata.obs_names].values
        tf_activity = tf_activity[order, :]

    adata.obsm["TF_activity"] = tf_activity
    adata.uns["TF_names"] = [str(x) for x in tf_names]

    print(f"Saving updated h5ad → {h5ad_out}")
    adata.write(h5ad_out)

    print(f"✓ Added TF activity for {len(tf_names)} TFs")
    return h5ad_out

# ==================================================
# STEP 7: DIFFERENTIAL TF ACTIVITY (Scanpy)
# ==================================================

def differential_tf_scanpy(h5ad_with_tf=OUTPUT_H5AD_WITH_TF, out_dir=OUTPUT_DIR):
    print("\n" + "="*60)
    print("Differential TF activity with Scanpy")
    print("="*60)

    adata = anndata.read_h5ad(h5ad_with_tf)

    # Merge Resistant + Aggressive → Aggressive
    rs = adata.obs["Response_Status"].astype(str).str.strip()
    merged = rs.copy()
    merged[rs.str.lower().isin(["resistant", "aggressive"])] = "Aggressive"
    merged[rs.str.lower().isin(["less-aggressive","less_aggressive","less aggressive"])] = "less-Aggressive"
    adata.obs["Response_Status_merged"] = pd.Categorical(merged)

    print("Merged groups:")
    print(adata.obs["Response_Status_merged"].value_counts())

    # TF AnnData
    tf_adata = sc.AnnData(
        X=adata.obsm["TF_activity"],
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=[str(x) for x in adata.uns["TF_names"]])
    )

    print("\nRunning rank_genes_groups(wilcoxon)...")
    sc.tl.rank_genes_groups(tf_adata, groupby="Response_Status_merged", method="wilcoxon")

    results = sc.get.rank_genes_groups_df(tf_adata, group=None)

    out_csv = os.path.join(out_dir, "tf_differential_activity_merged.csv")
    results.to_csv(out_csv, index=False)
    print(f"✓ Saved differential CSV: {out_csv}")

    return tf_adata, results, out_csv

# ==================================================
# STEP 8: VOLCANO PLOT
# ==================================================

def volcano_plot(tf_adata, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # Compute log2FC Aggressive vs less-Aggressive
    groups = tf_adata.obs["Response_Status_merged"]
    mask_agg  = (groups == "Aggressive").values
    mask_less = (groups == "less-Aggressive").values

    X = tf_adata.X
    if sparse.issparse(X):
        X = X.toarray()

    mean_agg  = X[mask_agg].mean(axis=0)
    mean_less = X[mask_less].mean(axis=0)
    log2fc = np.log2((mean_agg + 1) / (mean_less + 1))

    # p-values
    pvals = tf_adata.uns["rank_genes_groups"]["pvals"]["Aggressive"]
    pvals = np.array(pvals, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(log2fc, -np.log10(pvals + 1e-300), alpha=0.5, s=30)

    ax.axhline(-np.log10(0.05), ls="--", alpha=0.5, color='red', label='p=0.05')
    ax.axvline(0, ls="--", alpha=0.3)
    ax.set_xlabel("log2FC (Aggressive / less-Aggressive)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Differential TF Activity")
    ax.legend()

    out_png = os.path.join(out_dir, "volcano_tf_scanpy_merged.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"✓ Saved volcano plot: {out_png}")
    return out_png

# ==================================================
# MAIN
# ==================================================

def main():
    print("\n" + "="*60)
    print("TF ACTIVITY PIPELINE - FIXED VERSION")
    print("="*60)

    # Use existing deduped file
    fimo_path = DEDUP_FIMO_PATH

    # 1) peak mapping
    coord_to_idx, peak_ranges = create_peak_mapping(FASTA_FILE)

    # 2) load data
    X, obs, var = load_h5ad_manual(H5AD_PATH)
    obs = fix_response_status(obs)
    fimo = load_fimo_results(fimo_path)

    # 3) map fimo → peaks
    mapped_fimo = map_fimo_to_peaks(fimo, peak_ranges)
    if mapped_fimo is None:
        print("\n❌ No mapped FIMO hits. Check coordinates.")
        return

    # 4) TF activity
    tf_activity, tfs = create_tf_activity_matrix(X, mapped_fimo)
    if tf_activity is None:
        return

    # 5) save npz
    npz_path = save_tf_npz(tf_activity, tfs, obs.index.values)

    # 6) add to AnnData + save h5ad
    h5ad_with_tf = add_tf_to_anndata(npz_path)

    # 7) differential scanpy + volcano
    tf_adata, results, out_csv = differential_tf_scanpy(h5ad_with_tf)
    volcano_plot(tf_adata)

    print("\n✅ PIPELINE DONE!")
    print(f"\n📁 Outputs:")
    print(f"  - TF activity: {npz_path}")
    print(f"  - Updated h5ad: {h5ad_with_tf}")
    print(f"  - Differential results: {out_csv}")
    print(f"  - Volcano plot: {os.path.join(OUTPUT_DIR, 'volcano_tf_scanpy_merged.png')}")

if __name__ == "__main__":
    main()