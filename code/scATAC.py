import pandas as pd
import numpy as np
import os
import sys
import anndata
import scanpy as sc
from scipy.sparse import csr_matrix, diags
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import argparse
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
#                           TF-IDF NORMALIZATION
# ============================================================================

def tfidf_normalize(adata: anndata.AnnData) -> csr_matrix:
    """Apply TF-IDF normalization for scATAC-seq data."""
    print("Applying TF-IDF normalization...")
    
    X = adata.X.copy().tocsr()
    
    # Term Frequency: normalize by total counts per cell
    cell_sums = np.array(X.sum(axis=1)).flatten()
    cell_sums[cell_sums == 0] = 1
    tf = diags(1.0 / cell_sums).dot(X)
    
    # Inverse Document Frequency: weight by feature rarity
    n_cells = X.shape[0]
    feature_counts = np.array((X > 0).sum(axis=0)).flatten()
    feature_counts[feature_counts == 0] = 1
    idf = np.log(1 + n_cells / feature_counts)
    
    # Apply TF-IDF
    tfidf = tf.dot(diags(idf))
    
    print("✓ TF-IDF complete")
    return tfidf.tocsr()


# ============================================================================
#                        PEAK LOADING (MULTIPLE FORMATS)
# ============================================================================

def load_peaks(peaks_path: str) -> pd.DataFrame:
    """Load peak regions from various formats (BED, TSV with chr:start-end, chr_start_end)."""
    print(f"Loading peaks from: {peaks_path}")

    peaks_df = pd.read_csv(
        peaks_path,
        sep="\t",
        comment="#",
        header=None,
        skiprows=1,
        usecols=[0, 1, 2],
        low_memory=False
    )
    
    first_val = str(peaks_df.iloc[0, 0])
    
    # Format 1: 'chr1:100-200' in column 0
    if ":" in first_val and "-" in first_val:
        coords = peaks_df.iloc[:, 0].str.extract(r"(chr[^:]+):(\d+)-(\d+)")
        coords.columns = ["chromosome", "start", "end"]
        peaks_df = coords.astype({"start": int, "end": int})
    
    # Format 2: 'chr1_100_200' in column 0
    elif first_val.startswith('chr') and '_' in first_val:
        coords = peaks_df.iloc[:, 0].str.extract(r"(chr[^_]+)_(\d+)_(\d+)")
        coords.columns = ["chromosome", "start", "end"]
        peaks_df = coords.astype({"start": int, "end": int})
    
    # Format 3: Standard BED (chr, start, end in columns)
    else:
        try:
            peaks_df.columns = ["chromosome", "start", "end"]
        except ValueError:
            raise ValueError("Peak file format not recognized. Expected BED format or chr:start-end notation.")
    
    print("=== Peak head (after parsing) ===")
    print(peaks_df.head(5))
    
    # Clean and validate
    peaks_df['chromosome'] = peaks_df['chromosome'].astype(str)
    peaks_df['start'] = pd.to_numeric(peaks_df['start'], errors='coerce')
    peaks_df['end'] = pd.to_numeric(peaks_df['end'], errors='coerce')
    peaks_df = peaks_df.dropna()
    peaks_df['start'] = peaks_df['start'].astype(int)
    peaks_df['end'] = peaks_df['end'].astype(int)
    
    # Normalize chromosome naming (vectorized)
    mask = ~peaks_df['chromosome'].str.startswith('chr', na=False)
    peaks_df.loc[mask, 'chromosome'] = 'chr' + peaks_df.loc[mask, 'chromosome'].astype(str)
    
    # Create Feature_ID
    peaks_df['Feature_ID'] = (
        peaks_df['chromosome'] + ':' +
        peaks_df['start'].astype(str) + '-' +
        peaks_df['end'].astype(str)
    )
    
    peaks_df = peaks_df.sort_values(['chromosome', 'start']).reset_index(drop=True)
    print(f"✓ Loaded {len(peaks_df):,} peaks")
    
    return peaks_df


# ============================================================================
#                    GENERATE PEAKS FROM FRAGMENTS
# ============================================================================

def generate_peaks_from_fragments(
    fragments_path: str,
    min_cells: int = 3,
    min_peak_width: int = 20,
    max_peak_width: int = 10000,
    extend: int = 250,
    merge_buffer: int = 50,
    chunk_size: int = 1_000_000
) -> pd.DataFrame:
    """
    Generate a biologically relevant peak set from scATAC fragments.
    
    Mimics Cell Ranger / MACS-style peak calling:
      1. Load fragments
      2. Count distinct barcodes per genomic region
      3. Merge overlapping fragments into peaks
      4. Filter peaks by width (20–10,000 bp typical)
      5. Extend final peaks ±250 bp

    Args:
        fragments_path : str
            Path to fragments.tsv.gz from Cell Ranger ATAC
        min_cells : int
            Minimum number of distinct cells supporting a peak
        min_peak_width : int
            Minimum allowed peak width
        max_peak_width : int
            Maximum allowed peak width
        extend : int
            Extend each merged region on both sides
        merge_buffer : int
            Buffer distance for merging overlapping fragments (default: 50bp)
        chunk_size : int
            Read file in chunks to save memory
    """
    print("\n" + "=" * 60)
    print("GENERATING BIOLOGICAL PEAKS FROM FRAGMENTS")
    print("=" * 60)
    print(f"Input: {fragments_path}")
    print(f"Min cells per region: {min_cells}")
    print(f"Peak width range: {min_peak_width}-{max_peak_width} bp")
    print(f"Merge buffer: {merge_buffer} bp")
    print(f"Extension: ±{extend} bp")

    chrom_regions = defaultdict(list)
    total_frags = 0

    # Read fragments file in chunks
    reader = pd.read_csv(
        fragments_path,
        sep="\t",
        comment="#",
        header=None,
        compression="gzip",
        usecols=[0, 1, 2, 3],
        names=["chromosome", "start", "end", "barcode"],
        dtype={"chromosome": str, "start": int, "end": int, "barcode": str},
        chunksize=chunk_size,
        on_bad_lines="skip"
    )

    print("\nReading and grouping fragments...")
    for chunk in reader:
        # Normalize chromosome naming (vectorized)
        mask = ~chunk["chromosome"].str.startswith("chr", na=False)
        chunk.loc[mask, "chromosome"] = "chr" + chunk.loc[mask, "chromosome"].astype(str)
        
        for chrom, group in chunk.groupby("chromosome"):
            chrom_regions[chrom].extend(group[["start", "end", "barcode"]].values.tolist())
        total_frags += len(chunk)
    print(f"✓ Processed {total_frags:,} fragments total")

    # Merge overlapping regions and filter by barcode count
    print("\nMerging overlapping fragments into peaks...")
    peaks = []
    
    for chrom, regions in chrom_regions.items():
        # sort by start position
        regions.sort(key=lambda x: x[0])
        
        if not regions:
            continue
            
        merged_start, merged_end = regions[0][0], regions[0][1]
        barcodes = {regions[0][2]}

        for start, end, barcode in regions[1:]:
            if start <= merged_end + merge_buffer:  # overlap with small buffer
                merged_end = max(merged_end, end)
                barcodes.add(barcode)
            else:
                # Save current region if it passes filters
                if len(barcodes) >= min_cells:
                    peak_width = merged_end - merged_start
                    if min_peak_width <= peak_width <= max_peak_width:
                        peaks.append([chrom, 
                                      max(0, merged_start - extend), 
                                      merged_end + extend, 
                                      len(barcodes)])
                # start new region
                merged_start, merged_end = start, end
                barcodes = {barcode}

        # CRITICAL FIX: finalize last region for THIS chromosome
        if len(barcodes) >= min_cells:
            peak_width = merged_end - merged_start
            if min_peak_width <= peak_width <= max_peak_width:
                peaks.append([chrom, 
                             max(0, merged_start - extend), 
                             merged_end + extend, 
                             len(barcodes)])

    peaks_df = pd.DataFrame(peaks, columns=["chromosome", "start", "end", "n_cells"])
    peaks_df["Feature_ID"] = (
        peaks_df["chromosome"] + ":" +
        peaks_df["start"].astype(str) + "-" +
        peaks_df["end"].astype(str)
    )
    peaks_df.sort_values(["chromosome", "start"], inplace=True)
    peaks_df.reset_index(drop=True, inplace=True)

    print(f"✓ Generated {len(peaks_df):,} high-confidence peaks")
    print(f"  Median peak width: {(peaks_df['end'] - peaks_df['start']).median():.0f} bp")
    print(f"  Median cells per peak: {peaks_df['n_cells'].median():.0f}")

    return peaks_df[["chromosome", "start", "end", "Feature_ID"]]


# ============================================================================
#                      FRAGMENT MAPPING TO PEAKS
# ============================================================================

def map_fragments_to_peaks_efficient(fragments_path: str, 
                                     peaks_df: pd.DataFrame) -> pd.DataFrame:
    """Map fragments to peaks using interval overlap."""
    print(f"\nLoading fragments from: {fragments_path}")
    
    # Normalize chromosome naming in peaks_df (vectorized)
    mask = ~peaks_df["chromosome"].str.startswith("chr", na=False)
    peaks_df.loc[mask, "chromosome"] = "chr" + peaks_df.loc[mask, "chromosome"].astype(str)
    
    chunk_size = 1_000_000
    binned_chunks = []
    
    # --- Robust Fragment Loading (Handle 4-column files) ---
    try:
        # Attempt to read 5 columns
        fragments_reader = pd.read_csv(
            fragments_path,
            sep='\t',
            compression='gzip',
            comment='#',
            header=None,
            names=['chromosome', 'Start', 'End', 'Cell_Barcode', 'Count'],
            usecols=[0, 1, 2, 3, 4],
            dtype={'chromosome': str, 'Start': int, 'End': int, 'Cell_Barcode': str, 'Count': int},
            chunksize=chunk_size
        )
    except ValueError:
        # Fallback for 4-column fragments file
        print("Warning: Fragment file appears to be 4-column (chr, start, end, barcode). Assuming Count=1.")
        fragments_reader = pd.read_csv(
            fragments_path,
            sep='\t',
            compression='gzip',
            comment='#',
            header=None,
            names=['chromosome', 'Start', 'End', 'Cell_Barcode'],
            usecols=[0, 1, 2, 3],
            dtype={'chromosome': str, 'Start': int, 'End': int, 'Cell_Barcode': str},
            chunksize=chunk_size
        )

    for chunk_idx, fragments_chunk in enumerate(fragments_reader):
        if chunk_idx == 0:
            print(f"Processing fragments in chunks of {chunk_size:,}...")

        # If 4-column file was read (i.e., 'Count' is missing), add it back
        if 'Count' not in fragments_chunk.columns:
            fragments_chunk['Count'] = 1
        
        # Robust Chromosome Normalization (vectorized)
        mask = ~fragments_chunk["chromosome"].str.startswith("chr", na=False)
        fragments_chunk.loc[mask, "chromosome"] = "chr" + fragments_chunk.loc[mask, "chromosome"].astype(str)
        
        # Process per chromosome
        for chrom in fragments_chunk['chromosome'].unique():
            frags_chr = fragments_chunk[fragments_chunk['chromosome'] == chrom].copy()
            peaks_chr = peaks_df[peaks_df['chromosome'] == chrom]
            
            if peaks_chr.empty:
                continue
            
            # Sort for merge_asof
            frags_chr_sorted = frags_chr.sort_values('Start')
            peaks_chr_sorted = peaks_chr.sort_values('start')
            
            # Merge fragments to the closest (backward) peak start coordinate
            merged = pd.merge_asof(
                frags_chr_sorted,
                peaks_chr_sorted,
                left_on='Start',
                right_on='start',
                by='chromosome',
                direction='backward',
                suffixes=('_frag', '_peak')
            )
            
            # CRITICAL FIX: Check for ANY overlap between fragment and peak
            # A fragment overlaps a peak if:
            #   - Fragment end > peak start AND
            #   - Fragment start < peak end
            valid = (
                (merged['End'] > merged['start']) &
                (merged['Start'] < merged['end'])
            )
            
            if valid.any():
                binned_chunks.append(
                    merged[valid][['Feature_ID', 'Cell_Barcode', 'Count']]
                )
        
        if (chunk_idx + 1) % 10 == 0:
            print(f"  Processed {(chunk_idx + 1) * chunk_size:,} fragments...")
    
    if not binned_chunks:
        raise ValueError("No fragments overlapped with peaks! Check chromosome naming and file format.")
    
    binned_df = pd.concat(binned_chunks, ignore_index=True)
    print(f"✓ Mapped {len(binned_df):,} fragment-peak overlaps")
    
    binned_df = binned_df.groupby(['Feature_ID', 'Cell_Barcode'], as_index=False)['Count'].sum()
    print(f"✓ Aggregated to {len(binned_df):,} unique entries")
    
    return binned_df


# ============================================================================
#                          CREATE ANNDATA OBJECT
# ============================================================================

def create_anndata(binned_df: pd.DataFrame, 
                   peaks_df: pd.DataFrame, 
                   metadata_df: pd.DataFrame = None) -> anndata.AnnData:
    """Create AnnData object from binned fragment counts."""
    print("\nCreating sparse matrix...")
    
    unique_features = peaks_df['Feature_ID'].values
    unique_cells = binned_df['Cell_Barcode'].unique()
    
    feature_to_idx = {f: i for i, f in enumerate(unique_features)}
    cell_to_idx = {c: i for i, c in enumerate(unique_cells)}
    
    row_indices = binned_df['Feature_ID'].map(feature_to_idx).values
    col_indices = binned_df['Cell_Barcode'].map(cell_to_idx).values
    data = binned_df['Count'].values
    
    sparse_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(unique_features), len(unique_cells))
    )
    
    print(f"✓ Created sparse matrix: {sparse_matrix.shape} (Features × Cells)")
    
    # Create obs DataFrame
    if metadata_df is not None:
        obs_df = pd.DataFrame(index=unique_cells).join(metadata_df, how='left')
        matched = obs_df.notna().any(axis=1).sum()
        print(f"✓ Matched {matched:,}/{len(unique_cells):,} cells with metadata")
    else:
        obs_df = pd.DataFrame(index=unique_cells)
    
    # Create AnnData (Transpose to Cells × Features)
    adata = anndata.AnnData(
        X=sparse_matrix.T,
        obs=obs_df,
        var=peaks_df.set_index('Feature_ID')
    )
    
    return adata


# ============================================================================
#                    MINIMAL PREPROCESSING (QC ONLY)
# ============================================================================

def minimal_preprocess_atac(adata: anndata.AnnData,
                            min_genes: int = 50,
                            min_cells: int = 1,
                            apply_tfidf: bool = True,
                            use_sklearn_tfidf: bool = False) -> anndata.AnnData:
    """
    Minimal preprocessing pipeline - QC and normalization ONLY.
    NO PCA, NO clustering, NO UMAP.
    
    This preserves maximum information for downstream integration.
    """
    print("\n" + "="*60)
    print("MINIMAL PREPROCESSING (QC + NORMALIZATION ONLY)")
    print("="*60)
    
    # Store raw counts
    print("\n[1/4] Storing raw counts...")
    adata.layers['raw_counts'] = adata.X.copy()
    print("✓ Raw counts saved in layers['raw_counts']")
    
    # [2] Binarization (more efficient)
    print("\n[2/4] Binarization...")
    adata.X = (adata.X > 0).astype(np.int8)
    if not sp.issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    adata.layers['binary'] = adata.X.copy()
    print("✓ Binary matrix created and saved in layers['binary']")
    
    # [3] QC Metrics and Filtering
    print(f"\n[3/4] Quality control (min_genes={min_genes}, min_cells={min_cells})...")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    print(f"Before: {adata.n_obs:,} cells × {adata.n_vars:,} features")
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"After:  {adata.n_obs:,} cells × {adata.n_vars:,} features")
    
    # [4] TF-IDF (Optional)
    if apply_tfidf:
        print("\n[4/4] TF-IDF normalization...")
        if use_sklearn_tfidf:
            transformer = TfidfTransformer()
            adata.X = transformer.fit_transform(adata.X)
            sc.pp.normalize_total(adata, target_sum=1e4)
            print("✓ sklearn TF-IDF + normalization complete")
        else:
            adata.X = tfidf_normalize(adata)
            print("✓ Custom TF-IDF complete")
        adata.layers['tfidf'] = adata.X.copy()
    else:
        print("\n[4/4] Skipping TF-IDF normalization")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE - NO DIMENSIONALITY REDUCTION")
    print("="*60)
    print(f"Final shape: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")
    print("\nLayers available:")
    for layer in adata.layers.keys():
        print(f"  - {layer}")
    
    return adata


# ============================================================================
#                        MAIN PROCESSING FUNCTION
# ============================================================================

def process_fragments(fragments_path: str, 
                     peaks_path: str = None,
                     output_path: str = None, 
                     metadata_path: str = None,
                     generate_peaks: bool = False,
                     **kwargs):
    """
    Process fragments file (with or without peaks).
    
    Args:
        fragments_path: Path to fragments.tsv.gz
        peaks_path: Path to peaks file (optional if generate_peaks=True)
        output_path: Path for output .h5ad
        metadata_path: Path to cell metadata (optional)
        generate_peaks: If True, generate peaks from fragments
        **kwargs: Additional parameters
    """
    
    # Load or generate peaks
    if generate_peaks or peaks_path is None:
        peaks_df = generate_peaks_from_fragments(
            fragments_path,
            min_cells=kwargs.get('peak_min_cells', 3),
            min_peak_width=kwargs.get('min_peak_width', 20),
            max_peak_width=kwargs.get('max_peak_width', 10000),
            extend=kwargs.get('extend', 250),
            merge_buffer=kwargs.get('merge_buffer', 50)
        )
    else:
        peaks_df = load_peaks(peaks_path)
    
    # Load metadata if provided
    metadata_df = None
    if metadata_path:
        print(f"Loading metadata from: {metadata_path}")
        metadata_df = pd.read_csv(metadata_path, sep='\t', compression='gzip', index_col=0)
        print(f"✓ Loaded metadata for {len(metadata_df):,} cells")
    
    # Map fragments to peaks
    binned_df = map_fragments_to_peaks_efficient(fragments_path, peaks_df)
    
    # Create AnnData
    adata = create_anndata(binned_df, peaks_df, metadata_df)
    print(f"\nInitial AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")
    
    # Minimal preprocessing (QC only, no dimensionality reduction)
    adata = minimal_preprocess_atac(
        adata,
        min_genes=kwargs.get('min_genes', 50),
        min_cells=kwargs.get('min_cells', 1),
        apply_tfidf=kwargs.get('apply_tfidf', True),
        use_sklearn_tfidf=kwargs.get('use_sklearn_tfidf', False)
    )
    
    # Save
    if output_path:
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        adata.write_h5ad(output_path, compression='gzip')
        print(f"✓ Saved to: {output_path}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Final: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")
    print("\nThis h5ad contains:")
    print("  - Raw counts (layers['raw_counts'])")
    print("  - Binary matrix (layers['binary'])")
    if kwargs.get('apply_tfidf', True):
        print("  - TF-IDF normalized (layers['tfidf'] and .X)")
    print("  - QC metrics (obs and var)")
    print("\nReady for integration and downstream analysis!")
    
    return adata


# ============================================================================
#                               CLI INTERFACE
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="scATAC-seq Data Processor - Raw Data Preservation Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Fragments + peaks (BED format)
  python scATAC.py --fragments data.tsv.gz --peaks peaks.bed --output out.h5ad
  
  # Mode 2: Fragments + peaks (narrowPeak format)
  python scATAC.py --fragments data.tsv.gz --peaks peaks.narrowPeak --output out.h5ad
  
  # Mode 3: Fragments only (auto-generate peaks)
  python scATAC.py --fragments data.tsv.gz --generate-peaks --output out.h5ad
  
  # With metadata
  python scATAC.py --fragments data.tsv.gz --peaks peaks.bed --metadata meta.tsv.gz --output out.h5ad
        """
    )
    
    # Required arguments
    parser.add_argument('--fragments', required=True, help="Input fragments.tsv.gz")
    parser.add_argument('--output', required=True, help="Output .h5ad file")
    
    # Peak-related arguments
    parser.add_argument('--peaks', help="Input peaks file (BED/narrowPeak format)")
    parser.add_argument('--metadata', help="Cell metadata file (TSV)")
    parser.add_argument('--generate-peaks', action='store_true',
                       help="Generate peaks from fragments (no peak file needed)")
    
    # Peak generation parameters
    parser.add_argument('--peak-min-cells', type=int, default=3,
                       help="Minimum cells for peak generation (default: 3)")
    parser.add_argument('--min-peak-width', type=int, default=20,
                       help="Minimum peak width in bp (default: 20)")
    parser.add_argument('--max-peak-width', type=int, default=10000,
                       help="Maximum peak width in bp (default: 10000)")
    parser.add_argument('--extend', type=int, default=250,
                       help="Extension around peaks in bp (default: 250)")
    parser.add_argument('--merge-buffer', type=int, default=50,
                       help="Buffer distance for merging overlapping fragments in bp (default: 50)")
    
    # QC parameters (very conservative defaults for rare populations)
    parser.add_argument('--min-genes', type=int, default=50,
                       help="Minimum peaks per cell (default: 50, very lenient)")
    parser.add_argument('--min-cells', type=int, default=1,
                       help="Minimum cells per peak (default: 1, keep all peaks)")
    
    # Normalization options
    parser.add_argument('--no-tfidf', action='store_true',
                       help="Skip TF-IDF normalization")
    parser.add_argument('--sklearn-tfidf', action='store_true',
                       help="Use sklearn TF-IDF instead of custom")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.fragments):
        print(f"Error: Fragments file not found: {args.fragments}")
        sys.exit(1)
    
    if not args.generate_peaks and not args.peaks:
        parser.error("Either --peaks or --generate-peaks required")
    
    if args.peaks and not os.path.exists(args.peaks):
        print(f"Error: Peaks file not found: {args.peaks}")
        sys.exit(1)
    
    # Process
    kwargs = {
        'peak_min_cells': args.peak_min_cells,
        'min_peak_width': args.min_peak_width,
        'max_peak_width': args.max_peak_width,
        'extend': args.extend,
        'merge_buffer': args.merge_buffer,
        'min_genes': args.min_genes,
        'min_cells': args.min_cells,
        'apply_tfidf': not args.no_tfidf,
        'use_sklearn_tfidf': args.sklearn_tfidf,
    }
    
    process_fragments(
        fragments_path=args.fragments,
        peaks_path=args.peaks,
        output_path=args.output,
        metadata_path=args.metadata,
        generate_peaks=args.generate_peaks,
        **kwargs
    )


if __name__ == "__main__":
    main()