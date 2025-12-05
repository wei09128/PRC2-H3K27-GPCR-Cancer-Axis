#!/usr/bin/env python3
"""
Unified Single-Cell Pipeline with Gene Annotation
Batch processes scRNA-seq and scATAC-seq data with optional gene annotation

Usage:
  # scRNA-seq batch processing
  python scpipeline.py scrna --input_dir ./data --output_dir ./results --script_path scRNA.py
  
  # scATAC-seq batch processing  
  python scpipeline.py scatac --input_dir ./data --output_dir ./results --script_path scATAC.py
  
  # With gene annotation (scRNA only)
  python scpipeline.py scrna --input_dir ./data --output_dir ./results --annotate --gtf genes.gtf
"""

import argparse
import subprocess
import os
from pathlib import Path
import sys
import scanpy as sc
import anndata as ad
import pandas as pd
import gzip
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
#                         GENE ANNOTATION MODULE
# ============================================================================

class GeneAnnotator:
    """Annotate gene names in single-cell data using GTF files"""
    
    def __init__(self, gtf_path: str):
        self.gtf_path = Path(gtf_path)
        if not self.gtf_path.exists():
            raise FileNotFoundError(f"GTF file not found: {gtf_path}")
        self.gene_map = None
        print(f"Initializing GeneAnnotator with: {self.gtf_path.name}")
    
    def parse_gtf(self):
        """Parse GTF file to extract gene ID to gene name mapping"""
        print(f"\nParsing GTF file: {self.gtf_path}")
        gene_info = {}
        
        # Handle both compressed and uncompressed GTF files
        if self.gtf_path.suffix == '.gz':
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'
        
        with open_func(self.gtf_path, mode) as f:
            for i, line in enumerate(f):
                if i % 100000 == 0 and i > 0:
                    print(f"  Processed {i:,} lines, found {len(gene_info):,} genes...", end='\r')
                
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9 or parts[2] != 'gene':
                    continue
                
                # Parse attributes
                attributes = parts[8]
                attr_dict = {}
                
                for attr in attributes.split(';'):
                    attr = attr.strip()
                    if not attr:
                        continue
                    
                    if ' "' in attr:
                        key, value = attr.split(' "', 1)
                        value = value.rstrip('"')
                        attr_dict[key] = value
                
                gene_id = attr_dict.get('gene_id', '')
                gene_name = attr_dict.get('gene_name', '')
                
                if gene_id:
                    gene_info[gene_id] = {
                        'gene_name': gene_name if gene_name else gene_id,
                        'gene_type': attr_dict.get('gene_type', attr_dict.get('gene_biotype', '')),
                    }
        
        print(f"\n✓ Parsed GTF: found {len(gene_info):,} genes")
        return gene_info
    
    def load_gene_map(self):
        """Load gene mapping from GTF"""
        if self.gene_map is None:
            self.gene_map = self.parse_gtf()
    
    def annotate_adata(self, adata: ad.AnnData):
        """Annotate gene names in AnnData object IN-PLACE"""
        print(f"\nAnnotating AnnData: {adata.n_vars} genes")
        
        self.load_gene_map()
        
        # Auto-detect ID type
        sample_ids = adata.var_names[:100].tolist()
        is_ensembl = any(str(id).startswith('ENS') for id in sample_ids)
        
        # Store original IDs
        adata.var['original_id'] = adata.var_names.copy()
        
        if is_ensembl:
            print("  Detected: Ensembl IDs")
            # Remove version numbers
            clean_ids = [str(id).split('.')[0] for id in adata.var_names]
            
            new_names = []
            unmapped = 0
            for orig_id, clean_id in zip(adata.var_names, clean_ids):
                if clean_id in self.gene_map:
                    new_names.append(self.gene_map[clean_id]['gene_name'])
                    # Add gene_type annotation
                    adata.var.loc[orig_id, 'gene_type'] = self.gene_map[clean_id]['gene_type']
                else:
                    new_names.append(orig_id)
                    unmapped += 1
            
            adata.var_names = new_names
            adata.var_names_make_unique()
            print(f"  ✓ Mapped {len(new_names) - unmapped}/{len(new_names)} genes")
            if unmapped > 0:
                print(f"  ⚠ {unmapped} genes not found, kept original IDs")
        else:
            print("  Detected: Gene symbols (no mapping needed)")
        
        return adata


# ============================================================================
#                         BATCH PROCESSING MODULE
# ============================================================================

def batch_process_samples(input_dir, output_dir, script_path, modality_args, modality, annotate, gtf_path):
    """
    Scans input directory for MTX or fragment files and executes external script.
    Optionally annotates genes after processing (scRNA only).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print("\n" + "="*60)
    print(f"BATCH MODE: {modality.upper()}")
    print("="*60)
    print(f"Script: {script_path}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for ATAC fragment files first
    if modality == 'scatac':
        fragment_files = list(input_path.glob('*fragments.tsv.gz'))
        if fragment_files:
            print(f"\nDetected {len(fragment_files)} ATAC fragment file(s)")
            return batch_process_fragments(input_path, output_path, script_path, modality_args, fragment_files)
    
    # Otherwise process MTX files
    matrix_files = list(input_path.glob('*.mtx.gz'))
    
    if not matrix_files:
        if modality == 'scatac':
            print(f"\nERROR: No '*fragments.tsv.gz' or '*.mtx.gz' files found in {input_path}")
        else:
            print(f"\nERROR: No '*.mtx.gz' files found in {input_path}")
        return
    
    print(f"\nFound {len(matrix_files)} MTX sample(s)")
    
    # Build common arguments
    common_args = []
    for key, value in modality_args.items():
        arg_name = key.replace('_', '-')
        
        if isinstance(value, bool):
            if value:
                common_args.append(f"--{arg_name}")
        elif isinstance(value, list) and value:
            common_args.append(f"--{arg_name}")
            common_args.extend(str(item) for item in value)
        elif value is None:
            continue
        else:
            common_args.append(f"--{arg_name}")
            common_args.append(str(value))
    
    print(f"\nParameters: {' '.join(common_args)}")
    
    # Initialize annotator if needed
    annotator = None
    if annotate and modality == 'scrna':
        try:
            annotator = GeneAnnotator(gtf_path)
        except FileNotFoundError as e:
            print(f"⚠ WARNING: {e}")
            print("Skipping annotation...")
            annotator = None
    
    success_count = 0
    failed_count = 0

    for idx, matrix_file in enumerate(matrix_files, 1):
        prefix = matrix_file.name.removesuffix('.mtx.gz')
        
        # Find barcodes and features files
        barcode_candidates = [
            matrix_file.parent / f"{prefix}_barcodes.tsv.gz",
            matrix_file.parent / f"{prefix.replace('matrix', 'barcodes')}.tsv.gz",
        ]
        
        feature_candidates = [
            matrix_file.parent / f"{prefix}_features.tsv.gz",
            matrix_file.parent / f"{prefix.replace('matrix', 'features')}.tsv.gz",
        ]
        
        barcodes_file = next((f for f in barcode_candidates if f.exists()), None)
        features_file = next((f for f in feature_candidates if f.exists()), None)
        
        output_prefix = prefix.removesuffix('_matrix').removesuffix('-matrix').replace('_matrix-', '-')
        output_file = output_path / f"{output_prefix}.h5ad"

        if not (barcodes_file and features_file):
            print(f"\n[{idx}/{len(matrix_files)}] SKIPPED: {prefix}")
            print(f"  Reason: Missing barcodes or features file")
            failed_count += 1
            continue
        
        print(f"\n[{idx}/{len(matrix_files)}] Processing: {prefix}")
        print(f"  Matrix:   {matrix_file.name}")
        print(f"  Barcodes: {barcodes_file.name}")
        print(f"  Features: {features_file.name}")
        print(f"  Output:   {output_file.name}")

        # Build command
        command = [
            sys.executable, script_path,
            *common_args,
            "--matrix", str(matrix_file),
            "--barcodes", str(barcodes_file),
            "--features", str(features_file),
            "--output", str(output_file)
        ]
        
        # Execute command
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  ✓ SUCCESS: Generated {output_file.name}")
            
            # Annotate if requested (scRNA only)
            if annotator and output_file.exists():
                print(f"  📝 Annotating genes...")
                try:
                    adata = sc.read_h5ad(str(output_file))
                    adata = annotator.annotate_adata(adata)
                    adata.write_h5ad(str(output_file))
                    print(f"  ✓ Annotation complete")
                except Exception as e:
                    print(f"  ⚠ Annotation failed: {str(e)}")
            
            success_count += 1

        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED: Return code {e.returncode}")
            print(f"  Error: {e.stderr[-300:].strip()}")
            failed_count += 1
        except FileNotFoundError:
            print(f"  ✗ FAILED: Script '{script_path}' not found")
            failed_count += 1

        print("-" * 60)
    
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"✓ Successful: {success_count}/{len(matrix_files)}")
    print(f"✗ Failed:     {failed_count}/{len(matrix_files)}")
    print("="*60 + "\n")


def batch_process_fragments(input_path, output_path, script_path, modality_args, fragment_files):
    """Process multiple ATAC fragment files in batch mode"""
    
    # Build common arguments for fragment mode
    common_args = []
    for key, value in modality_args.items():
        arg_name = key.replace('_', '-')
        
        if isinstance(value, bool):
            if value:
                common_args.append(f"--{arg_name}")
        elif isinstance(value, list) and value:
            common_args.append(f"--{arg_name}")
            common_args.extend(str(item) for item in value)
        elif value is None:
            continue
        else:
            common_args.append(f"--{arg_name}")
            common_args.append(str(value))
    
    # Add generate-peaks flag
    common_args.append("--generate-peaks")
    
    print(f"\nParameters: {' '.join(common_args)}")

    success_count = 0
    failed_count = 0

    for idx, fragment_file in enumerate(fragment_files, 1):
        sample_name = fragment_file.name.replace('_fragments.tsv.gz', '').replace('fragments.tsv.gz', '')
        output_file = output_path / f"{sample_name}.h5ad"
        
        print(f"\n[{idx}/{len(fragment_files)}] Processing: {sample_name}")
        print(f"  Fragment: {fragment_file.name}")
        print(f"  Output:   {output_file.name}")

        # Build command
        command = [
            sys.executable, script_path,
            *common_args,
            "--fragments", str(fragment_file),
            "--output", str(output_file)
        ]
        
        # Execute
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  ✓ SUCCESS: Generated {output_file.name}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED: Return code {e.returncode}")
            print(f"  Error: {e.stderr[-300:].strip()}")
            failed_count += 1
        except FileNotFoundError:
            print(f"  ✗ FAILED: Script '{script_path}' not found")
            failed_count += 1
        
        print("-" * 60)
    
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"✓ Successful: {success_count}/{len(fragment_files)}")
    print(f"✗ Failed:     {failed_count}/{len(fragment_files)}")
    print("="*60 + "\n")


# ============================================================================
#                         ARGUMENT PARSER SETUP
# ============================================================================

def setup_argument_parser():
    """Sets up the argument parser with all parameters"""
    parser = argparse.ArgumentParser(
        description="Unified single-cell pipeline for batch processing with optional gene annotation",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # scRNA-seq batch (keep all genes, lenient filtering)
  python scpipeline.py scrna \\
    --input_dir ./data \\
    --output_dir ./results \\
    --script_path scRNA.py \\
    --min_genes 100 \\
    --min_cells 1 \\
    --n_top_genes 50000
  
  # scRNA-seq with annotation
  python scpipeline.py scrna \\
    --input_dir ./data \\
    --output_dir ./results \\
    --annotate \\
    --gtf GRCh38.gtf
  
  # scATAC-seq batch (fragment files)
  python scpipeline.py scatac \\
    --input_dir ./data \\
    --output_dir ./results \\
    --script_path scATAC.py \\
    --min_genes 50 \\
    --min_cells 1
  
  # scATAC-seq with custom peak parameters
  python scpipeline.py scatac \\
    --input_dir ./data \\
    --output_dir ./results \\
    --peak_min_cells 3 \\
    --extend 250
        """
    )

    # Modality selection (required)
    parser.add_argument('modality', choices=['scrna', 'scatac'],
                       help='Data modality to process')

    # Batch configuration (required)
    batch_group = parser.add_argument_group('Batch Configuration')
    batch_group.add_argument('--input_dir', type=str, required=True,
                            help='Input directory containing data files')
    batch_group.add_argument('--output_dir', type=str, default="./output",
                            help='Output directory for .h5ad files (default: ./output)')
    batch_group.add_argument('--script_path', type=str, default=None,
                            help='Path to processing script (auto-detected if not specified)')

    # Gene annotation (scRNA only)
    annotation_group = parser.add_argument_group('Gene Annotation (scRNA only)')
    annotation_group.add_argument('--annotate', action='store_true',
                                 help='Annotate gene names using GTF file')
    annotation_group.add_argument('--gtf', type=str, default='GCF_000001405.26_GRCh38_genomic.gtf',
                                 help='GTF file for annotation (default: GCF_000001405.26_GRCh38_genomic.gtf)')

    # Common parameters
    common_group = parser.add_argument_group('Quality Control Parameters')
    common_group.add_argument('--min_genes', type=int, default=None,
                             help='Minimum features per cell (default: RNA=100, ATAC=50)')
    common_group.add_argument('--min_cells', type=int, default=1,
                             help='Minimum cells per feature (default: 1, very lenient)')

    # scRNA-seq specific
    rna_group = parser.add_argument_group('scRNA-seq Parameters')
    rna_group.add_argument('--target_sum', type=float, default=1e4,
                          help='Normalization target sum (default: 10000)')
    rna_group.add_argument('--n_top_genes', type=int, default=None,
                          help='Number of highly variable genes (default: None, keep all)')
    rna_group.add_argument('--mt_prefix', type=str, default='mt-',
                          help='Mitochondrial gene prefix (default: "mt-", use "MT-" for human)')
    rna_group.add_argument('--no_normalization', action='store_true',
                          help='Skip normalization and log-transformation (NOT RECOMMENDED)')

    # scATAC-seq specific
    atac_group = parser.add_argument_group('scATAC-seq Parameters')
    atac_group.add_argument('--peak_min_cells', type=int, default=3,
                           help='Minimum cells for peak calling (default: 3)')
    atac_group.add_argument('--min_peak_width', type=int, default=20,
                           help='Minimum peak width in bp (default: 20)')
    atac_group.add_argument('--max_peak_width', type=int, default=10000,
                           help='Maximum peak width in bp (default: 10000)')
    atac_group.add_argument('--extend', type=int, default=250,
                           help='Extension around peaks in bp (default: 250)')
    atac_group.add_argument('--no_tfidf', action='store_true',
                           help='Skip TF-IDF normalization')
    atac_group.add_argument('--sklearn_tfidf', action='store_true',
                           help='Use sklearn TF-IDF instead of custom')
    
    return parser


# ============================================================================
#                              MAIN FUNCTION
# ============================================================================

def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Extract configuration
    all_kwargs = vars(args)
    input_dir = all_kwargs.pop('input_dir')
    output_dir = all_kwargs.pop('output_dir')
    script_path = all_kwargs.pop('script_path')
    modality = all_kwargs.pop('modality')
    annotate = all_kwargs.pop('annotate')
    gtf_path = all_kwargs.pop('gtf')

    # Auto-detect script path
    if not script_path:
        script_path = f"sc{'ATAC' if modality == 'scatac' else 'RNA'}.py"
        print(f"Auto-detected script: {script_path}")

    # Set modality-specific defaults
    if all_kwargs['min_genes'] is None:
        all_kwargs['min_genes'] = 100 if modality == 'scrna' else 50

    # Build parameter dictionary for subprocess
    if modality == 'scrna':
        relevant_keys = [
            'min_genes', 'min_cells', 'target_sum', 
            'n_top_genes', 'mt_prefix', 'no_normalization'
        ]
    elif modality == 'scatac':
        relevant_keys = [
            'min_genes', 'min_cells', 'peak_min_cells',
            'min_peak_width', 'max_peak_width', 'extend',
            'no_tfidf', 'sklearn_tfidf'
        ]
    
    relevant_params = {k: v for k, v in all_kwargs.items() if k in relevant_keys}
    
    # Process batch
    batch_process_samples(
        input_dir, 
        output_dir, 
        script_path, 
        relevant_params, 
        modality,
        annotate,
        gtf_path
    )


if __name__ == '__main__':
    main()