import anndata as ad
import pandas as pd
from pyfaidx import Fasta
import os

# --- USER PARAMETERS (UPDATED) ---
H5AD_PATH = '/mnt/f/H3K27/Data/scATAC/scATAC_peaks_QC_BC_float64_clean.h5ad'
# CRITICAL: This path is now CORRECTLY pointing to the FASTA genome file.
GENOME_FASTA_PATH = '/mnt/f/H3K27/hg38.fa' 
OUTPUT_FASTA_PATH = '/mnt/f/H3K27/Data/TF/scATAC_peak_sequences.fasta'

def extract_peak_sequences(h5ad_path: str, fasta_path: str, output_fasta: str):
    """
    Step 1: Extracts DNA sequences for all accessible chromatin peaks defined in 
    the AnnData object's .var_names (assuming they are in 'chr:start-end' format).
    The extracted sequences are saved to a FASTA file for subsequent motif analysis.
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
        print(f"Error loading FASTA: {e}. Ensure the file exists and is a valid FASTA format.")
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
            
            # --- Chromosome Name Check ---
            # If the peak coordinate has a 'chr' prefix (e.g., 'chr1') but the FASTA 
            # file uses names without it (e.g., '1'), we need to try both.
            chrom_name_options = [chrom]
            if chrom.startswith('chr'):
                chrom_name_options.append(chrom.replace('chr', ''))
            elif not chrom.startswith('chr'):
                chrom_name_options.append('chr' + chrom)
            
            sequence = None
            for name in chrom_name_options:
                if name in genome:
                    # pyfaidx is 1-based inclusive for start, 1-based inclusive for end.
                    # Sequence extraction is [start:end] where start and end are 1-based positions.
                    # The pyfaidx documentation suggests [start-1:end] for 0-based start, 1-based end coordinates.
                    # Assuming .var_names uses 1-based coordinates:
                    sequence = str(genome[name][start - 1 : end])
                    break

            if sequence is None:
                raise KeyError(f"Chromosome {chrom} not found in FASTA.")

            # Format output in standard FASTA format: >peak_id \n sequence
            output_lines.append(f">{peak_id}\n{sequence.upper()}")
            
        except KeyError:
            # Handles cases where the chromosome might not exist in the FASTA (e.g., scaffolds or non-standard chroms)
            failed_peaks += 1
        except ValueError:
            # Handles cases where the format is incorrect (not chr:start-end)
            failed_peaks += 1
        except Exception as e:
            # Catch other unexpected errors
            # print(f"Error processing peak {peak_id}: {e}") # Uncomment for verbose debugging
            failed_peaks += 1
            
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
            
    # Write all sequences to the output file
    if output_lines:
        with open(output_fasta, 'w') as f:
            f.write('\n'.join(output_lines) + '\n')
        print(f"\nSuccessfully extracted sequences for {len(peak_coords) - failed_peaks} peaks.")
        print(f"{failed_peaks} peaks failed (likely due to chromosome naming mismatch or missing sequence).")
        print(f"Output saved to: {output_fasta}")
    else:
        print("\nNo sequences were successfully extracted. Check your AnnData file and FASTA header names.")


# --- EXECUTION ---
if __name__ == '__main__':
    # You must have successfully downloaded the hg38.fa file to /mnt/f/H3K27/
    extract_peak_sequences(H5AD_PATH, GENOME_FASTA_PATH, OUTPUT_FASTA_PATH)