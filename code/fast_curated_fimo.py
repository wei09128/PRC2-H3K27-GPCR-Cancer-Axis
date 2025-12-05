#!/usr/bin/env python3
# fast_curated_fimo.py - Optimized parallel FIMO for FASTA input

import subprocess
import multiprocessing as mp
from pathlib import Path
import sys

def run_command(cmd, description="", exit_on_error=True):
    """Run shell command with error handling"""
    if description:
        print(f"  → {description}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and exit_on_error:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    return result.stdout

def split_fasta(input_fasta, outdir, n_chunks):
    """Split FASTA file into equal chunks"""
    
    print(f"\n[1/2] Splitting FASTA into {n_chunks} chunks...")
    
    # Count sequences
    n_seqs = int(run_command(f"grep -c '^>' {input_fasta}").strip())
    print(f"  Total sequences: {n_seqs:,}")
    
    seqs_per_chunk = (n_seqs // n_chunks) + 1
    print(f"  ~{seqs_per_chunk:,} sequences per chunk")
    
    # Use awk to split by sequence count
    awk_script = f"""
    BEGIN {{ chunk=0; count=0; file="{outdir}/chunk_00.fa" }}
    /^>/ {{ 
        if (count >= {seqs_per_chunk}) {{ 
            close(file);
            chunk++;
            count=0;
            file=sprintf("{outdir}/chunk_%02d.fa", chunk);
        }}
        count++;
    }}
    {{ print > file }}
    """
    
    run_command(f"awk '{awk_script}' {input_fasta}", "Splitting sequences")
    
    # Get chunk files
    chunks = sorted(outdir.glob("chunk_*.fa"))
    print(f"  ✓ Created {len(chunks)} chunk files")
    
    return chunks

def run_fimo_chunk(args):
    """Run FIMO on a single chunk"""
    chunk_file, chunk_id, motifs, outdir = args
    
    try:
        # Run FIMO with optimized settings
        output_file = outdir / f"fimo_{chunk_id}.txt"
        cmd = f"""
        fimo --thresh 1e-5 \
             --max-strand \
             --verbosity 1 \
             --max-stored-scores 5000000 \
             --text \
             {motifs} \
             {chunk_file} \
             > {output_file} 2>&1
        """
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Check if output has content (FIMO returns 0 even with no hits)
        if not Path(output_file).exists() or Path(output_file).stat().st_size == 0:
            return f"⚠ Chunk {chunk_id} - no hits found"
        
        # Count hits (exclude header)
        n_hits = int(subprocess.run(
            f"tail -n +2 {output_file} | wc -l",
            shell=True, capture_output=True, text=True
        ).stdout.strip())
        
        return f"✓ Chunk {chunk_id} complete ({n_hits:,} hits)"
        
    except Exception as e:
        return f"✗ Chunk {chunk_id} failed: {e}"

def combine_results(outdir, n_chunks):
    """Combine all FIMO results into one file"""
    
    print("\n[3/3] Combining results...")
    
    combined_file = outdir / "fimo_combined.txt"
    
    # Find first valid result file
    first_file = None
    for i in range(n_chunks):
        test_file = outdir / f"fimo_{i:02d}.txt"
        if test_file.exists() and test_file.stat().st_size > 0:
            first_file = test_file
            break
    
    if first_file is None:
        print("WARNING: No FIMO results found!")
        return None, 0
    
    # Write header from first file
    with open(first_file) as f:
        header = f.readline()
    
    with open(combined_file, 'w') as out:
        out.write(header)
        
        # Append data from all files (skip headers)
        files_combined = 0
        for i in range(n_chunks):
            result_file = outdir / f"fimo_{i:02d}.txt"
            if result_file.exists() and result_file.stat().st_size > 0:
                with open(result_file) as f:
                    f.readline()  # Skip header
                    content = f.read()
                    if content.strip():  # Only write if has content
                        out.write(content)
                        files_combined += 1
    
    print(f"  ✓ Combined {files_combined} result files")
    
    # Count total hits
    total_hits = int(run_command(f"tail -n +2 {combined_file} | wc -l").strip())
    
    return combined_file, total_hits

def main():
    print("=" * 60)
    print("  Fast Parallel FIMO Analysis")
    print("=" * 60)
    
    # Configuration
    INPUT_FASTA = "/mnt/f/H3K27/Data/TF/scATAC_peak_sequences.fasta"
    MOTIFS = "/mnt/f/H3K27/Data/TF/FIMO_curated/curated_motifs.meme"
    OUTDIR = Path("/mnt/f/H3K27/Data/TF/fimo_parallel")
    
    # Verify files exist
    if not Path(INPUT_FASTA).exists():
        print(f"ERROR: Input FASTA not found: {INPUT_FASTA}")
        sys.exit(1)
    
    if not Path(MOTIFS).exists():
        print(f"ERROR: Motif file not found: {MOTIFS}")
        
        # Try to find it
        alt_paths = [
            "/mnt/f/H3K27/Data/TF/curated_motifs.meme",
            "/mnt/f/H3K27/Data/TF/motifs.meme"
        ]
        
        for alt in alt_paths:
            if Path(alt).exists():
                MOTIFS = alt
                print(f"  → Found motifs at: {MOTIFS}")
                break
        else:
            print("\nCould not find motif file. Please check path.")
            sys.exit(1)
    
    print(f"\nInput FASTA: {INPUT_FASTA}")
    print(f"Motif file:  {MOTIFS}")
    
    # Create output directory
    OUTDIR.mkdir(exist_ok=True)
    print(f"Output dir:  {OUTDIR}")
    
    # Get number of cores (cap at 16 for efficiency)
    n_cores = min(mp.cpu_count(), 16)
    print(f"Using cores: {n_cores}")
    
    # Step 1: Split FASTA
    chunks = split_fasta(INPUT_FASTA, OUTDIR, n_cores)
    
    if len(chunks) == 0:
        print("ERROR: Failed to create chunks!")
        sys.exit(1)
    
    # Step 2: Run FIMO in parallel
    print(f"\n[2/3] Running FIMO on {len(chunks)} cores in parallel...")
    print("  (Estimated time: 30-90 minutes)")
    print("")
    
    # Prepare arguments
    args_list = [
        (chunk, f"{i:02d}", MOTIFS, OUTDIR)
        for i, chunk in enumerate(chunks)
    ]
    
    # Run in parallel
    with mp.Pool(len(chunks)) as pool:
        results = pool.map(run_fimo_chunk, args_list)
    
    # Print results
    for result in results:
        print(f"  {result}")
    
    # Step 3: Combine results
    combined_file, total_hits = combine_results(OUTDIR, len(chunks))
    
    if combined_file is None:
        print("\n⚠ WARNING: No motif hits found!")
        print("This might indicate:")
        print("  - Motif file is incompatible")
        print("  - Sequences are too short")
        print("  - Threshold is too stringent")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Total motif hits:  {total_hits:,}")
    print(f"Output file:       {combined_file}")
    print(f"Average per seq:   {total_hits / int(run_command('grep -c \"^>\" ' + INPUT_FASTA).strip()):.1f}")
    print("")
    print("Next steps:")
    print("  1. Analyze motif enrichment per TF")
    print("  2. Map motifs back to peak regions")
    print("  3. Identify TF-peak associations")
    print("=" * 60)

if __name__ == "__main__":
    # Check for FIMO
    result = subprocess.run("which fimo", shell=True, capture_output=True)
    if result.returncode != 0:
        print("ERROR: FIMO not found in PATH")
        print("Please activate your meme_env:")
        print("  conda activate meme_env")
        sys.exit(1)
    
    main()