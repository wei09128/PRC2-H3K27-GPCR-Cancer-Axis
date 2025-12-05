import anndata as ad
import pandas as pd
from pyfaidx import Fasta
import os

# --- USER PARAMETERS ---
H5AD_PATH = 'scATAC_peaks_QC_BC_float64_clean.h5ad'
GENOME_FASTA_PATH = '/path/to/your/reference_genome.fa' # e.g., 'hg38.fa' or 'mm10.fa'
OUTPUT_FASTA_PATH = 'scATAC_peak_sequences.fasta'

def extract_peak_sequences(h5ad_path: str, fasta_path: str, output_fasta: str):
    """
    Step 1: Extracts DNA sequences for all peaks defined in the AnnData object's 
    .var_names (assuming they are in 'chr:start-end' format).
    """
    print(f"Loading AnnData object from: {h5ad_path}")
    try:
        adata = ad.read_h5ad(h5ad_path)
    except FileNotFoundError:
        print(f"Error: AnnData file not found at {h5ad_path}")
        return
    except Exception as e:
        print(f"Error loading AnnData: {e}")
        return

    print(f"Loading reference genome from: {fasta_path}")
    try:
        # Load the genome FASTA file using pyfaidx for fast access
        genome = Fasta(fasta_path)
    except FileNotFoundError:
        print(f"Error: Genome FASTA file not found at {fasta_path}")
        return
    except Exception as e:
        print(f"Error loading FASTA: {e}")
        return

    peak_coords = adata.var_names
    output_lines = []
    
    # Track errors
    failed_peaks = 0
    
    print(f"Total peaks to process: {len(peak_coords)}")
    print("Extracting sequences...")

    for peak_id in peak_coords:
        try:
            # Peak ID should be in the format 'chrX:start-end' (e.g., 'chr1:1000-2000')
            chrom, rest = peak_id.split(':')
            start, end = map(int, rest.split('-'))
            
            # pyfaidx extracts sequences using 1-based indexing for start (start) 
            # and 1-based indexing for end (end).
            # Note: FASTA sequence names are typically without the 'chr' prefix 
            # if your genome FASTA is formatted that way. You might need to adjust 'chrom'.
            
            sequence = str(genome[chrom][start - 1 : end])
            
            # Format output in standard FASTA format: >peak_id \n sequence
            output_lines.append(f">{peak_id}\n{sequence.upper()}")
            
        except KeyError:
            # This handles cases where the chromosome might not exist in the FASTA (e.g., scaffolds)
            failed_peaks += 1
        except ValueError:
            # This handles cases where the format is incorrect (not chr:start-end)
            failed_peaks += 1
        except Exception as e:
            failed_peaks += 1
            
    # Write all sequences to the output file
    with open(output_fasta, 'w') as f:
        f.write('\n'.join(output_lines) + '\n')

    print(f"\nSuccessfully extracted sequences for {len(peak_coords) - failed_peaks} peaks.")
    print(f"{failed_peaks} peaks failed (likely due to missing FASTA entry or incorrect format).")
    print(f"Output saved to: {output_fasta}")


# --- EXECUTION ---
if __name__ == '__main__':
    # You must update H5AD_PATH and GENOME_FASTA_PATH before running
    extract_peak_sequences(H5AD_PATH, GENOME_FASTA_PATH, OUTPUT_FASTA_PATH)