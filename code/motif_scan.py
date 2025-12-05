#!/usr/bin/env python3
"""
Fast motif scanning using MOODS (5-10x faster than FIMO)
Requires: pip install MOODS-python biopython pandas --break-system-packages
"""

import MOODS.scan
import MOODS.tools
import MOODS.parsers
from Bio import SeqIO
import pandas as pd
from pathlib import Path
import numpy as np

def scan_motifs_moods(fasta_file, meme_file, output_file, pvalue_threshold=1e-4):
    """
    Scan sequences for motif matches using MOODS
    
    Parameters:
    -----------
    fasta_file : str
        Path to FASTA file with peak sequences
    meme_file : str
        Path to MEME format motif file
    output_file : str
        Output TSV file path
    pvalue_threshold : float
        P-value threshold for matches (default: 1e-4)
    """
    
    print(f"Loading motifs from {meme_file}...")
    motifs, motif_names = MOODS.parsers.pfm_to_log_odds(
        meme_file, 
        pseudocount=0.001
    )
    
    # Calculate thresholds for each motif
    print(f"Calculating thresholds for p-value {pvalue_threshold}...")
    bg = MOODS.tools.flat_bg(4)  # Equal background
    thresholds = [
        MOODS.tools.threshold_from_p(m, bg, pvalue_threshold) 
        for m in motifs
    ]
    
    # Initialize scanner
    scanner = MOODS.scan.Scanner(7)  # window size
    scanner.set_motifs(motifs, bg, thresholds)
    
    # Scan sequences
    print(f"Scanning sequences from {fasta_file}...")
    results = []
    
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        if i % 1000 == 0:
            print(f"Processed {i} sequences...")
        
        seq = str(record.seq).upper()
        hits = scanner.scan(seq)
        
        # Process hits for each motif
        for motif_idx, motif_hits in enumerate(hits):
            for hit in motif_hits:
                strand = '+' if hit.pos >= 0 else '-'
                position = abs(hit.pos)
                score = hit.score
                
                results.append({
                    'motif_id': motif_names[motif_idx],
                    'sequence_name': record.id,
                    'start': position,
                    'stop': position + len(motifs[motif_idx][0]),
                    'strand': strand,
                    'score': score,
                    'p-value': pvalue_threshold,  # Approximate
                    'matched_sequence': seq[position:position + len(motifs[motif_idx][0])]
                })
    
    # Create DataFrame and save
    print(f"Found {len(results)} motif matches")
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        df = df.sort_values(['sequence_name', 'start'])
    
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Results saved to {output_file}")
    
    return df


def scan_motifs_simple(fasta_file, meme_file, output_file, pvalue_threshold=1e-4):
    """
    Simplified version if MOODS has issues
    Uses regex-based scanning (slower but no dependencies)
    """
    from Bio import motifs
    from Bio.Seq import Seq
    import re
    
    print(f"Loading motifs from {meme_file}...")
    with open(meme_file) as f:
        meme_motifs = motifs.parse(f, 'meme')
    
    print(f"Scanning sequences from {fasta_file}...")
    results = []
    
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        if i % 100 == 0:
            print(f"Processed {i} sequences...")
        
        seq = str(record.seq).upper()
        
        for motif in meme_motifs:
            pwm = motif.counts.normalize(pseudocounts=0.5)
            pssm = pwm.log_odds()
            
            # Scan forward strand
            for position, score in pssm.search(Seq(seq), threshold=7.0):
                results.append({
                    'motif_id': motif.name,
                    'sequence_name': record.id,
                    'start': position,
                    'stop': position + len(motif),
                    'strand': '+',
                    'score': score,
                    'matched_sequence': seq[position:position + len(motif)]
                })
            
            # Scan reverse strand
            for position, score in pssm.reverse_complement().search(Seq(seq), threshold=7.0):
                results.append({
                    'motif_id': motif.name,
                    'sequence_name': record.id,
                    'start': position,
                    'stop': position + len(motif),
                    'strand': '-',
                    'score': score,
                    'matched_sequence': seq[position:position + len(motif)]
                })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Found {len(results)} motif matches. Results saved to {output_file}")
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python fast_motif_scan.py <fasta> <meme> <output> [pvalue]")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    meme_file = sys.argv[2]
    output_file = sys.argv[3]
    pvalue = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
    
    try:
        # Try MOODS first (faster)
        df = scan_motifs_moods(fasta_file, meme_file, output_file, pvalue)
    except ImportError:
        print("MOODS not available, using BioPython (slower)...")
        df = scan_motifs_simple(fasta_file, meme_file, output_file, pvalue)
    
    print(f"\nDone! Found {len(df)} matches across {df['sequence_name'].nunique()} sequences")