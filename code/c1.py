#!/usr/bin/env python3
"""
pre_flight_check.py

Check all files and data structures before running the TF pipeline.
"""

import os
import pandas as pd
import h5py
import numpy as np

# ==================================================
# CONFIG (match your pipeline)
# ==================================================

H5AD_PATH = "/mnt/f/H3K27/Data/scATAC/scATAC_peaks_QC_BC_float64_clean.h5ad"
FASTA_FILE = "/mnt/f/H3K27/Data/TF/scATAC_peak_sequences.fasta"
FIMO_DEDUP = "/mnt/f/H3K27/Data/TF/fimo_parallel/fimo_dedup.txt"
FIMO_COMBINED = "/mnt/f/H3K27/Data/TF/fimo_parallel/fimo_partial_combined.txt"

def check_file_exists(filepath, description):
    """Check if file exists and print size"""
    print(f"\n{'='*60}")
    print(f"Checking: {description}")
    print(f"{'='*60}")
    print(f"Path: {filepath}")
    
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"✓ EXISTS - Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ NOT FOUND")
        return False

def check_h5ad(filepath):
    """Check h5ad structure"""
    if not check_file_exists(filepath, "H5AD File"):
        return
    
    try:
        with h5py.File(filepath, "r") as f:
            print("\n📊 H5AD Structure:")
            
            # Check X
            if isinstance(f["X"], h5py.Dataset):
                print(f"  X: Dense array, shape {f['X'].shape}")
            else:
                shape = tuple(f["X"].attrs["shape"])
                print(f"  X: Sparse matrix, shape {shape}")
                print(f"     Cells: {shape[0]:,}")
                print(f"     Peaks: {shape[1]:,}")
            
            # Check obs
            if "obs" in f:
                print(f"\n  obs keys ({len(f['obs'].keys())} columns):")
                for key in list(f["obs"].keys())[:20]:  # Show first 20
                    print(f"    - {key}")
                if len(f["obs"].keys()) > 20:
                    print(f"    ... and {len(f['obs'].keys()) - 20} more")
                
                # Check Response_Status specifically
                if "Response_Status" in f["obs"]:
                    rs_data = f["obs"]["Response_Status"][:]
                    rs_decoded = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in rs_data]
                    print(f"\n  Response_Status values:")
                    unique, counts = np.unique(rs_decoded, return_counts=True)
                    for val, cnt in zip(unique, counts):
                        print(f"    {val}: {cnt:,}")
                else:
                    print(f"\n  ⚠️ Response_Status not found in obs")
            
            # Check if cell IDs exist
            if "_index" in f["obs"]:
                idx_data = f["obs"]["_index"][:]
                print(f"\n  Cell IDs: {len(idx_data):,}")
                print(f"    First 3: {[x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in idx_data[:3]]}")
            
    except Exception as e:
        print(f"❌ Error reading h5ad: {e}")

def check_fasta(filepath):
    """Check FASTA structure"""
    if not check_file_exists(filepath, "FASTA File"):
        return
    
    try:
        print("\n📊 FASTA Structure:")
        
        with open(filepath, "r") as f:
            # Count sequences and check format
            seq_count = 0
            coord_examples = []
            
            for i, line in enumerate(f):
                if line.startswith(">"):
                    seq_count += 1
                    if seq_count <= 5:
                        coord_examples.append(line.strip()[1:])
                
                if i > 100000:  # Sample first 100k lines
                    break
        
        print(f"  Total sequences (peaks): {seq_count:,}")
        print(f"\n  Coordinate format examples:")
        for ex in coord_examples:
            print(f"    {ex}")
        
        # Detect coordinate style
        if coord_examples:
            if ":" in coord_examples[0]:
                print(f"\n  ✓ Coordinate style: chr:start-end")
            else:
                print(f"\n  ⚠️ Unusual coordinate style detected")
        
    except Exception as e:
        print(f"❌ Error reading FASTA: {e}")

def check_fimo_file(filepath, description):
    """Check FIMO file structure"""
    if not check_file_exists(filepath, description):
        return
    
    try:
        print("\n📊 FIMO File Structure:")
        
        # Read first few lines to check format
        with open(filepath, "r") as f:
            lines = [f.readline() for _ in range(10)]
        
        print(f"\n  First 10 lines:")
        for i, line in enumerate(lines, 1):
            print(f"    {i}: {line.strip()[:100]}")
        
        # Try to read with pandas
        print(f"\n  Attempting pandas read...")
        
        # Try different read strategies
        try:
            # Strategy 1: Standard read
            df = pd.read_csv(filepath, sep="\t", nrows=100)
            print(f"  ✓ Read successful (standard)")
            print(f"\n  Columns: {list(df.columns)}")
            print(f"  Shape (first 100 rows): {df.shape}")
            
        except Exception as e1:
            print(f"  ❌ Standard read failed: {e1}")
            
            try:
                # Strategy 2: Skip comments, no header inference
                df = pd.read_csv(filepath, sep="\t", comment="#", nrows=100)
                print(f"  ✓ Read successful (comment skip)")
                print(f"\n  Columns: {list(df.columns)}")
                print(f"  Shape (first 100 rows): {df.shape}")
                
            except Exception as e2:
                print(f"  ❌ Comment skip read failed: {e2}")
                
                try:
                    # Strategy 3: No header, manual column names
                    column_names = ['motif_id', 'motif_alt_id', 'sequence_name', 
                                  'start', 'stop', 'strand', 'score', 
                                  'p-value', 'q-value', 'matched_sequence']
                    df = pd.read_csv(filepath, sep="\t", comment="#", 
                                   header=None, names=column_names, nrows=100)
                    print(f"  ✓ Read successful (manual columns)")
                    print(f"\n  Columns: {list(df.columns)}")
                    print(f"  Shape (first 100 rows): {df.shape}")
                    
                except Exception as e3:
                    print(f"  ❌ Manual columns read failed: {e3}")
                    return
        
        # Show data sample
        if df is not None:
            print(f"\n  Sample data:")
            print(df.head(3).to_string())
            
            # Check for key columns
            print(f"\n  Column checks:")
            for col in ['motif_alt_id', 'sequence_name', 'start', 'stop', 'score']:
                if col in df.columns:
                    print(f"    ✓ {col} found")
                    if col in ['start', 'stop', 'score']:
                        print(f"      dtype: {df[col].dtype}, nulls: {df[col].isna().sum()}")
                else:
                    print(f"    ❌ {col} MISSING")
            
            # Check coordinate format
            if 'sequence_name' in df.columns:
                sample_seqs = df['sequence_name'].head(5).tolist()
                print(f"\n  Sequence name examples:")
                for seq in sample_seqs:
                    print(f"    {seq}")
                
                has_colon = any(":" in str(s) for s in sample_seqs)
                if has_colon:
                    print(f"  ✓ Coordinate style: chr:start-end")
                else:
                    print(f"  ✓ Coordinate style: chr only (genomic coords)")
        
    except Exception as e:
        print(f"❌ Error checking FIMO file: {e}")

def main():
    print("\n" + "="*60)
    print("PRE-FLIGHT CHECK FOR TF ACTIVITY PIPELINE")
    print("="*60)
    
    # Check all files
    check_h5ad(H5AD_PATH)
    check_fasta(FASTA_FILE)
    check_fimo_file(FIMO_COMBINED, "FIMO Combined File")
    check_fimo_file(FIMO_DEDUP, "FIMO Deduplicated File")
    
    print("\n" + "="*60)
    print("PRE-FLIGHT CHECK COMPLETE")
    print("="*60)
    print("\nRecommendations:")
    print("1. If FIMO files show column issues, the deduplication step may need fixing")
    print("2. Check if coordinate styles match between FASTA and FIMO")
    print("3. Verify Response_Status values match expected groups")
    print("4. Ensure all files are accessible and not corrupted")

if __name__ == "__main__":
    main()