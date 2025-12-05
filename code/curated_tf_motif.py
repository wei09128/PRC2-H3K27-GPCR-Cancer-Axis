#!/usr/bin/env python3
"""
Extract curated TF motifs from JASPAR and run optimized FIMO
Focused on H3K27me3/PRC2 biology and immune/stress signatures
"""

import subprocess
import re
from pathlib import Path

# Configuration
JASPAR_MEME = "/mnt/f/H3K27/JASPAR_Vertebrates_nonredundant.meme"
FASTA_FILE = "/mnt/f/H3K27/Data/TF/scATAC_peak_sequences.fasta"
OUTPUT_DIR = "/mnt/f/H3K27/Data/TF/FIMO_curated"
CURATED_MEME = f"{OUTPUT_DIR}/curated_motifs.meme"

# Curated TF list - focused on PRC2/H3K27me3 biology
CURATED_TFS = {
    # PRC2 recruitment
    "YY1", "REST", "JARID2", "MTF2",
    
    # HOX genes (PRC2 canonical targets)
    "HOXA1", "HOXA2", "HOXA3", "HOXA5", "HOXA9",
    "HOXB1", "HOXB2", "HOXB4", "HOXB13",
    "HOXC9", "HOXD3", "HOXD9",
    
    # Pluripotency & development
    "SOX2", "SOX9", "NANOG", "POU5F1", "OCT4",
    "KLF4", "FOXO1", "FOXO3", "FOXA1",
    
    # NF-κB & inflammation
    "RELA", "NFKB1", "REL", "NFKB", "NFKB1", "NFKB2",
    
    # Interferon response  
    "IRF1", "IRF2", "IRF3", "IRF4", "IRF8",
    "STAT1", "STAT3", "STAT5A", "STAT5B",
    
    # Immune differentiation
    "SPI1", "PU.1", "ETS1", "RUNX1", "GATA3",
    
    # AP-1 & stress (matches gene signature)
    "FOS", "FOSB", "JUN", "JUNB", "JUND",
    "ATF3", "ATF4", "EGR1", "EGR2", "NR4A1",
    
    # Tumor suppressors
    "TP53", "TP63", "TP73", "E2F1",
    
    # Metabolism & hypoxia
    "HIF1A", "EPAS1", "PPARG", "PPARA",
    "NR1D2", "BHLHE41", "MYC", "MYCN",
    
    # Cancer-associated
    "AR", "ESR1", "ESR2",
    "SNAI1", "SNAI2", "TWIST1", "ZEB1",
    "GATA4", "TCF7", "TCF7L2", "LEF1",
}

def extract_motifs():
    """Extract curated motifs from JASPAR MEME file"""
    
    print("=" * 60)
    print("Extracting Curated TF Motifs")
    print("=" * 60)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Read JASPAR file
    with open(JASPAR_MEME, 'r') as f:
        content = f.read()
    
    # Split into motif blocks
    motifs = re.split(r'\nMOTIF ', content)
    header = motifs[0]
    motifs = ['MOTIF ' + m for m in motifs[1:]]
    
    # Extract matching motifs
    selected_motifs = []
    found_tfs = set()
    
    for motif_block in motifs:
        # Extract TF name from motif block
        # Format: MOTIF MA0139.1 CTCF
        match = re.search(r'MOTIF\s+(\S+)\s+(\S+)', motif_block)
        if match:
            motif_id = match.group(1)
            tf_name = match.group(2).upper()
            
            # Check if this TF is in our curated list
            if any(curated.upper() in tf_name for curated in CURATED_TFS):
                selected_motifs.append(motif_block)
                found_tfs.add(tf_name)
                print(f"  ✓ Found: {motif_id} {tf_name}")
    
    # Write curated MEME file
    with open(CURATED_MEME, 'w') as f:
        f.write(header)
        f.write('\n')
        for motif in selected_motifs:
            f.write(motif)
            if not motif.endswith('\n'):
                f.write('\n')
    
    print(f"\n{'=' * 60}")
    print(f"Extracted {len(selected_motifs)} motifs")
    print(f"Covering {len(found_tfs)} TFs")
    print(f"Output: {CURATED_MEME}")
    print(f"{'=' * 60}\n")
    
    return len(selected_motifs)

def run_fimo():
    """Run FIMO with curated motifs on ALL peaks"""
    
    print("=" * 60)
    print("Running Optimized FIMO")
    print("=" * 60)
    
    # Use all peaks (no filtering)
    input_fasta = FASTA_FILE
    
    # Count sequences
    result = subprocess.run([
        "grep", "-c", "^>", input_fasta
    ], capture_output=True, text=True, check=True)
    n_seqs = int(result.stdout.strip())
    
    print(f"Using ALL peaks: {n_seqs:,} sequences")
    print(f"\nRunning FIMO (estimated 3-5 hours with curated motifs)...")
    print(f"  Motif database: {CURATED_MEME}")
    print(f"  Sequences: {input_fasta}")
    print(f"  Output: {OUTPUT_DIR}/fimo_results.tsv")
    print()
    
    # Run FIMO
    cmd = [
        "fimo",
        "--oc", OUTPUT_DIR,
        "--thresh", "1e-4",
        "--text",
        "--no-qvalue",
        "--verbosity", "1",
        CURATED_MEME,
        input_fasta
    ]
    
    with open(f"{OUTPUT_DIR}/fimo_results.tsv", 'w') as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, check=True)
    
    # Count results
    result = subprocess.run([
        "wc", "-l", f"{OUTPUT_DIR}/fimo_results.tsv"
    ], capture_output=True, text=True, check=True)
    n_matches = int(result.stdout.split()[0]) - 1  # Subtract header
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Total motif matches: {n_matches:,}")
    print(f"Results: {OUTPUT_DIR}/fimo_results.tsv")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    try:
        n_motifs = extract_motifs()
        run_fimo()
        
        print("\n✅ SUCCESS!")
        print(f"\nNext steps:")
        print(f"1. Load results into Python/R")
        print(f"2. Map motifs to your gene signatures")
        print(f"3. Analyze TF binding patterns in H3K27me3 regions")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()