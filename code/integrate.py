#!/usr/bin/env python3
"""
Optimized Single-Cell Integration Pipeline
- Preserves layers['raw_counts'] through gene collapse
- MT genes filtered ONLY during CNV (not in main pipeline)
- Keeps all cells including G0 (200-300 counts)
- Memory-efficient operations
- Removes mm10_ genes and collapses hg19_ prefixes
"""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import harmonypy as hpy
from scipy.sparse import csr_matrix, csc_matrix, issparse
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import sys
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
import psutil
import gc
import scipy.sparse as sp
import gffutils
import os

warnings.filterwarnings('ignore')
sc.settings.verbosity = 3
sc.set_figure_params(facecolor="white")


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def normalize_gene_name(name):
    """
    Remove species prefix (mm10_, hg19_) and version suffix
    Returns: (cleaned_name, species)
    """
    species = 'human'  # default
    
    if name.startswith('mm10_'):
        species = 'mouse'
        name = name[5:]  # Remove 'mm10_'
    elif name.startswith('hg19_'):
        species = 'human'
        name = name[5:]  # Remove 'hg19_'
    
    # Remove version suffix (.1, .2, etc)
    base = name.split('.')[0]
    
    # Remove sub-copy suffix (-1, -2)
    if '-' in base and base.split('-')[-1].isdigit():
        base = '-'.join(base.split('-')[:-1])
    
    return base, species


def collapse_genes_preserve_layers(adata, remove_mouse=True, verbose=True):
    """
    Collapse genes (remove mm10, merge hg19_ duplicates) while preserving ALL layers
    This is MEMORY-EFFICIENT - operates on sparse matrices directly
    """
    print(f"\n{'='*70}")
    print("GENE COLLAPSE WITH FULL LAYER PRESERVATION")
    print(f"{'='*70}")
    print(f"Initial: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Check what layers exist
    layer_names = list(adata.layers.keys())
    print(f"Layers to preserve: {layer_names}")
    has_raw_counts = 'raw_counts' in layer_names
    
    # 1. Build gene mapping
    print("\nBuilding gene mapping...")
    gene_map = {}
    species_map = {}
    
    for gene in adata.var_names:
        clean_gene, species = normalize_gene_name(gene)
        gene_map[gene] = clean_gene
        species_map[gene] = species
    
    # Count statistics
    n_mouse = sum(1 for s in species_map.values() if s == 'mouse')
    n_human = sum(1 for s in species_map.values() if s == 'human')
    
    print(f"  Human genes: {n_human:,}")
    print(f"  Mouse genes: {n_mouse:,}")
    
    # 2. Filter mouse genes if requested
    if remove_mouse and n_mouse > 0:
        print(f"\n→ Removing {n_mouse:,} mouse genes...")
        keep_mask = np.array([species_map[g] == 'human' for g in adata.var_names])
        adata = adata[:, keep_mask].copy()
        
        # Update mappings
        gene_map = {k: v for k, v in gene_map.items() if species_map[k] == 'human'}
        print(f"  After removal: {adata.n_vars:,} genes")
    
    # 3. Group genes by collapsed name
    from collections import defaultdict
    collapse_groups = defaultdict(list)
    
    for gene in adata.var_names:
        target = gene_map[gene]
        collapse_groups[target].append(gene)
    
    unique_genes = sorted(collapse_groups.keys())
    n_duplicates = sum(1 for v in collapse_groups.values() if len(v) > 1)
    
    print(f"\n→ Collapsing to {len(unique_genes):,} unique genes")
    print(f"  Genes with duplicates: {n_duplicates:,}")
    
    # 4. Build collapse matrix (sparse!)
    print("\nBuilding collapse matrix...")
    row_idx = []
    col_idx = []
    
    for new_idx, new_gene in enumerate(unique_genes):
        for old_gene in collapse_groups[new_gene]:
            old_idx = list(adata.var_names).index(old_gene)
            row_idx.append(old_idx)
            col_idx.append(new_idx)
    
    collapse_matrix = csr_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)),
        shape=(adata.n_vars, len(unique_genes))
    )
    print(f"  Collapse matrix: {collapse_matrix.shape}")
    
    # 5. Collapse .X
    print("\nCollapsing .X...")
    if issparse(adata.X):
        X_collapsed = adata.X @ collapse_matrix
    else:
        X_collapsed = adata.X @ collapse_matrix.toarray()
    
    print(f"  ✓ X collapsed: {X_collapsed.shape}")
    
    # 6. Collapse ALL layers (CRITICAL!)
    layers_collapsed = {}
    
    for layer_name in layer_names:
        print(f"\nCollapsing layers['{layer_name}']...")
        layer_data = adata.layers[layer_name]
        
        if issparse(layer_data):
            layers_collapsed[layer_name] = layer_data @ collapse_matrix
        else:
            layers_collapsed[layer_name] = layer_data @ collapse_matrix.toarray()
        
        # Verify integrity for raw_counts
        if layer_name == 'raw_counts':
            sample = layers_collapsed[layer_name][:100, :100]
            if hasattr(sample, 'toarray'):
                sample = sample.toarray()
            
            mean_val = sample.mean()
            max_val = layers_collapsed[layer_name].max()
            is_int = np.allclose(sample, np.round(sample))
            
            print(f"  ✓ raw_counts collapsed")
            print(f"    Mean: {mean_val:.4f}, Max: {max_val:.1f}, Integers: {is_int}")
            
            if not is_int or mean_val < 0.1:
                print(f"    ⚠ WARNING: raw_counts may be corrupted!")
        else:
            print(f"  ✓ {layer_name} collapsed")
    
    # 7. Collapse .raw if it exists
    raw_collapsed = None
    if adata.raw is not None:
        print(f"\nCollapsing .raw...")
        # Build mapping for raw genes
        raw_gene_map = {}
        for gene in adata.raw.var_names:
            clean_gene, species = normalize_gene_name(gene)
            if not remove_mouse or species == 'human':
                raw_gene_map[gene] = clean_gene
        
        # Build collapse matrix for raw
        raw_collapse_groups = defaultdict(list)
        for gene in adata.raw.var_names:
            if gene in raw_gene_map:
                target = raw_gene_map[gene]
                raw_collapse_groups[target].append(gene)
        
        raw_unique_genes = sorted(raw_collapse_groups.keys())
        
        row_idx = []
        col_idx = []
        for new_idx, new_gene in enumerate(raw_unique_genes):
            for old_gene in raw_collapse_groups[new_gene]:
                old_idx = list(adata.raw.var_names).index(old_gene)
                row_idx.append(old_idx)
                col_idx.append(new_idx)
        
        raw_collapse_matrix = csr_matrix(
            (np.ones(len(row_idx)), (row_idx, col_idx)),
            shape=(adata.raw.n_vars, len(raw_unique_genes))
        )
        
        if issparse(adata.raw.X):
            raw_X_collapsed = adata.raw.X @ raw_collapse_matrix
        else:
            raw_X_collapsed = adata.raw.X @ raw_collapse_matrix.toarray()
        
        raw_collapsed = ad.AnnData(
            X=raw_X_collapsed,
            var=pd.DataFrame(index=raw_unique_genes)
        )
        print(f"  ✓ .raw collapsed: {raw_collapsed.n_vars:,} genes")
    
    # 8. Create new AnnData with all collapsed data
    print(f"\nCreating collapsed AnnData...")
    adata_collapsed = ad.AnnData(
        X=X_collapsed,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=unique_genes),
        layers=layers_collapsed,  # ← CRITICAL
        uns=adata.uns.copy() if adata.uns else {},
        obsm=adata.obsm.copy() if adata.obsm else {},
        obsp=adata.obsp.copy() if adata.obsp else {}
    )
    
    if raw_collapsed is not None:
        adata_collapsed.raw = raw_collapsed
    
    print(f"\n{'='*50}")
    print("COLLAPSE COMPLETE:")
    print(f"  Cells: {adata_collapsed.n_obs:,}")
    print(f"  Genes: {adata_collapsed.n_vars:,}")
    print(f"  Layers preserved:")
    for layer_name in layers_collapsed.keys():
        print(f"    ✓ {layer_name}")
    if adata_collapsed.raw:
        print(f"  .raw genes: {adata_collapsed.raw.n_vars:,}")
    print(f"{'='*50}\n")
    
    return adata_collapsed


def get_db(gtf_path, db_path):
    """Load or create gffutils database"""
    if not os.path.exists(db_path):
        print(f"Creating database from GTF: {gtf_path}...")
        db = gffutils.create_db(
            gtf_path, 
            dbfn=db_path, 
            force=True, 
            keep_order=True, 
            merge_strategy='merge',
            disable_infer_transcripts=True, 
            disable_infer_genes=True,
            id_spec={'gene': 'gene_id'}
        )
    else:
        print(f"Loading database from: {db_path}")
        db = gffutils.FeatureDB(db_path)
    return db


class SingleCellIntegrator:
    """Memory-efficient integration pipeline"""
    
    GRCH38_GTF_PATH = '/mnt/f/H3K27/gencode.v47.annotation.gtf'
    GRCH38_DB_PATH = '/mnt/f/H3K27/Data/gencode_v47.db'
    
    def __init__(
        self,
        mode: str = 'scRNA',
        n_top_genes: int = 2000,
        n_comps: int = 50,
        reduction_method: str = 'pca',
        max_scale_value: float = 10,
        harmony_max_iter: int = 100,
        batch_key: str = 'Condition',
        leiden_resolution: float = 0.5,
        min_cells_frac: float = 0.0001,
        plot_dir: Optional[str] = None,
        use_incremental: bool = False,
        apply_scaling: bool = False,
        collapse_genes: bool = True,
        remove_mouse: bool = True,
        annotate_genes: bool = True
    ):
        self.mode = mode.upper()
        self.n_top_genes = n_top_genes
        self.n_comps = n_comps
        self.reduction_method = reduction_method
        self.max_scale_value = max_scale_value
        self.harmony_max_iter = harmony_max_iter
        self.batch_key = batch_key
        self.leiden_resolution = leiden_resolution
        self.min_cells_frac = min_cells_frac
        self.plot_dir = plot_dir
        self.use_incremental = use_incremental
        self.apply_scaling = apply_scaling
        self.collapse_genes = collapse_genes
        self.remove_mouse = remove_mouse
        self.annotate_genes = annotate_genes
        
        if self.mode not in ['SCRNA', 'SCATAC']:
            raise ValueError(f"Mode must be 'scRNA' or 'scATAC', got {mode}")
    
    def annotate_with_genomic_coords(self, adata):
        """Add chromosome coordinates from GTF"""
        print(f"\nAnnotating with GRCh38 coordinates...")
        
        try:
            grch38_db = get_db(self.GRCH38_GTF_PATH, self.GRCH38_DB_PATH)
        except Exception as e:
            print(f"  ⚠ Could not load GTF database: {e}")
            print(f"  → Skipping genomic annotation")
            return adata
        
        # Build gene index
        gene_index = {}
        for gene in grch38_db.features_of_type('gene'):
            if 'gene_name' in gene.attributes:
                for name in gene.attributes['gene_name']:
                    chrom = str(gene.seqid).replace('chr', '')
                    if chrom and (chrom.isdigit() or chrom.lower() in ['x', 'y', 'mt', 'm']):
                        gene_index[name] = {
                            'chromosome': chrom,
                            'start': gene.start,
                            'end': gene.end
                        }
        
        print(f"  GTF database: {len(gene_index):,} genes indexed")
        
        # Map to current genes
        gene_coordinates = {}
        for gene_name in adata.var_names:
            if gene_name in gene_index:
                gene_coordinates[gene_name] = gene_index[gene_name]
        
        # Add to .var
        coords_df = pd.DataFrame.from_dict(gene_coordinates, orient='index')
        adata.var = adata.var.merge(coords_df, left_index=True, right_index=True, how='left')
        
        n_mapped = (~adata.var['chromosome'].isna()).sum() if 'chromosome' in adata.var else 0
        print(f"  ✓ Mapped {n_mapped:,}/{adata.n_vars:,} genes ({100*n_mapped/adata.n_vars:.1f}%)")
        
        # Filter unmapped genes
        if 'chromosome' in adata.var.columns and n_mapped > 0:
            initial_genes = adata.n_vars
            genes_to_keep = adata.var.index[~adata.var['chromosome'].isna()]
            adata = adata[:, genes_to_keep].copy()
            print(f"  Removed {initial_genes - adata.n_vars:,} unmapped genes")
            print(f"  Remaining: {adata.n_vars:,} genes")
        
        return adata
    
    def load_h5ad_files(self, input_path: str, pattern: str = "*.h5ad") -> Dict[str, ad.AnnData]:
        """Load h5ad files and identify raw counts"""
        print(f"\n{'='*70}")
        print("STEP 1: LOADING DATA")
        print(f"{'='*70}")
        print(f"Memory: {get_memory_usage():.2f} GB")
        
        input_dir = Path(input_path)
        h5ad_files = list(input_dir.glob(pattern))
        
        if not h5ad_files:
            raise FileNotFoundError(f"No h5ad files found in {input_path}")
        
        print(f"Found {len(h5ad_files)} files")
        
        adatas = {}
        for file_path in h5ad_files:
            condition = file_path.stem.replace('PROCESSED_', '').replace('_singlets', '')
            
            adata_temp = sc.read_h5ad(str(file_path))
            adata_temp.var_names_make_unique()
            
            print(f"\n  {condition}:")
            print(f"    Shape: {adata_temp.n_obs:,} × {adata_temp.n_vars:,}")
            print(f"    Layers: {list(adata_temp.layers.keys())}")
            
            # Identify raw counts (don't move yet!)
            raw_source = None
            if 'raw_counts' in adata_temp.layers:
                raw_source = 'layers[raw_counts]'
            elif 'counts' in adata_temp.layers:
                raw_source = 'layers[counts]'
            elif adata_temp.raw is not None:
                raw_source = '.raw'
            
            if raw_source:
                print(f"    ✓ Raw counts: {raw_source}")
            else:
                print(f"    ⚠ No raw counts found")
            
            # Set up .X for integration
            if self.mode == 'SCRNA':
                if 'log_normalized' in adata_temp.layers:
                    adata_temp.X = adata_temp.layers['log_normalized'].copy()
                    print(f"    → Using log_normalized for .X")
            elif self.mode == 'SCATAC':
                if 'tfidf' in adata_temp.layers:
                    adata_temp.X = adata_temp.layers['tfidf'].copy()
                    print(f"    → Using tfidf for .X")
            
            # Ensure sparse
            if not issparse(adata_temp.X):
                adata_temp.X = csr_matrix(adata_temp.X)
            
            adatas[condition] = adata_temp
        
        print(f"\nMemory after load: {get_memory_usage():.2f} GB")
        return adatas
    
    def concatenate_data(self, adatas: Dict[str, ad.AnnData]) -> ad.AnnData:
        """Concatenate with layer preservation"""
        print(f"\n{'='*70}")
        print("STEP 2: CONCATENATION")
        print(f"{'='*70}")
        
        # Concatenate
        adata = ad.concat(
            list(adatas.values()),
            join='outer',
            label=self.batch_key,
            keys=list(adatas.keys()),
            merge='unique'
        )
        
        adata.obs_names_make_unique()
        
        if not isinstance(adata.X, csr_matrix):
            adata.X = csr_matrix(adata.X)
        
        print(f"✓ Concatenated: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        print(f"  Batches: {adata.obs[self.batch_key].nunique()}")
        print(f"  Layers: {list(adata.layers.keys())}")
        
        # Check layer preservation
        if 'raw_counts' in adata.layers:
            sample = adata.layers['raw_counts'][:10, :10]
            if hasattr(sample, 'toarray'):
                sample = sample.toarray()
            print(f"  ✓ raw_counts preserved (mean: {sample.mean():.2f})")
        else:
            print(f"  ⚠ raw_counts layer NOT preserved!")
        
        del adatas
        gc.collect()
        
        return adata
    
    def preprocess_scrna(self, adata: ad.AnnData) -> ad.AnnData:
        """scRNA preprocessing - keeps all cells (including G0)"""
        print(f"\n{'='*70}")
        print("STEP 3: scRNA PREPROCESSING (NO CELL FILTERING)")
        print(f"{'='*70}")
        
        print(f"Input: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        print(f"  → Keeping cells including G0 with ~200-300 counts")
        
        # Filter genes with zero expression
        sc.pp.filter_genes(adata, min_cells=1)
        print(f"  Genes after zero-filter: {adata.n_vars:,}")
        
        # HVG selection
        print(f"\nSelecting {self.n_top_genes} HVGs...")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=self.n_top_genes,
            flavor='seurat_v3',
            subset=False
        )
        n_hvg = adata.var.highly_variable.sum()
        print(f"  ✓ {n_hvg} HVGs identified")
        
        # Subset to HVGs
        adata = adata[:, adata.var.highly_variable].copy()
        print(f"  ✓ Subsetted to {adata.n_vars:,} HVGs")
        
        # Optional scaling
        if self.apply_scaling:
            print(f"\nScaling (max={self.max_scale_value})...")
            sc.pp.scale(adata, max_value=self.max_scale_value)
        else:
            print("\nSkipping scaling")
        
        # PCA
        print(f"\nRunning PCA (n_comps={self.n_comps})...")
        if self.use_incremental or adata.n_obs > 1000000:
            adata = self.run_incremental_pca(adata, n_comps=self.n_comps)
        else:
            sc.tl.pca(adata, svd_solver='arpack', n_comps=self.n_comps)
        print(f"  ✓ PCA complete")
        
        gc.collect()
        return adata
    
    def preprocess_scatac(self, adata: ad.AnnData) -> ad.AnnData:
        """scATAC preprocessing with robust NaN handling"""
        print(f"\n{'='*70}")
        print("STEP 3: scATAC PREPROCESSING")
        print(f"{'='*70}")
        
        print(f"Input: {adata.n_obs:,} cells × {adata.n_vars:,} peaks")
        
        # Filter peaks
        min_cells = max(1, int(adata.n_obs * self.min_cells_frac))
        print(f"Filtering peaks in >= {min_cells} cells ({self.min_cells_frac*100:.4f}%)...")
        sc.pp.filter_genes(adata, min_cells=min_cells)
        print(f"  Remaining: {adata.n_vars:,} peaks")
        
        # Optional: Select top variable peaks
        if self.n_top_genes and self.n_top_genes < adata.n_vars:
            print(f"\nSelecting top {self.n_top_genes} variable peaks...")
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.n_top_genes,
                flavor='seurat_v3',
                subset=True
            )
            print(f"  ✓ Subsetted to {adata.n_vars:,} peaks")
        
        # Ensure sparse CSR format
        if not isinstance(adata.X, csr_matrix):
            print("  Converting to CSR format...")
            adata.X = csr_matrix(adata.X)
        
        print(f"  Matrix sparsity: {(1 - adata.X.nnz / (adata.n_obs * adata.n_vars)) * 100:.2f}%")
        
        # Check for problematic features before LSI
        print("\nChecking data quality...")
        
        # Remove features with zero variance
        feature_sums = np.array(adata.X.sum(axis=0)).flatten()
        zero_features = (feature_sums == 0)
        n_zero = zero_features.sum()
        
        if n_zero > 0:
            print(f"  Removing {n_zero:,} zero-sum features...")
            adata = adata[:, ~zero_features].copy()
            print(f"  Remaining: {adata.n_vars:,} peaks")
        
        # Check for cells with zero counts
        cell_sums = np.array(adata.X.sum(axis=1)).flatten()
        zero_cells = (cell_sums == 0)
        n_zero_cells = zero_cells.sum()
        
        if n_zero_cells > 0:
            print(f"  ⚠ WARNING: {n_zero_cells:,} cells have zero counts!")
            print(f"  Removing zero-count cells...")
            adata = adata[~zero_cells, :].copy()
            print(f"  Remaining: {adata.n_obs:,} cells")
        
        # LSI via TruncatedSVD with robust settings
        n_comps = min(self.n_comps, adata.n_vars - 1, adata.n_obs - 1)
        
        print(f"\nRunning LSI (TruncatedSVD, n_comps={n_comps})...")
        print(f"  Matrix shape: {adata.X.shape}")
        
        try:
            svd = TruncatedSVD(
                n_components=n_comps,
                random_state=0,
                algorithm='arpack',  # More stable than 'randomized' for sparse data
                n_iter=7  # Increased iterations for convergence
            )
            Z = svd.fit_transform(adata.X)
            
            print(f"  ✓ SVD complete")
            print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.3f}")
            
            # CRITICAL: Check for NaN/Inf BEFORE normalization
            n_nan = np.isnan(Z).sum()
            n_inf = np.isinf(Z).sum()
            
            if n_nan > 0 or n_inf > 0:
                print(f"  ⚠ Found {n_nan} NaN and {n_inf} Inf values in LSI")
                print(f"  → Replacing with zeros...")
                Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
            
            # L2 normalization (row-wise)
            print(f"  Applying L2 normalization...")
            row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
            
            # Check for zero-norm rows
            n_zero_norm = (row_norms.flatten() == 0).sum()
            if n_zero_norm > 0:
                print(f"  ⚠ Found {n_zero_norm} zero-norm rows")
                print(f"  → Adding small jitter...")
                # Add small jitter to zero rows
                zero_mask = (row_norms.flatten() == 0)
                Z[zero_mask] += np.random.RandomState(0).randn(n_zero_norm, n_comps) * 1e-10
                row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
            
            # Safe division
            Z = np.divide(Z, row_norms, out=np.zeros_like(Z), where=row_norms!=0)
            
            # Final NaN check after normalization
            n_nan_final = np.isnan(Z).sum()
            n_inf_final = np.isinf(Z).sum()
            
            if n_nan_final > 0 or n_inf_final > 0:
                print(f"  ⚠ Still have NaN/Inf after normalization!")
                print(f"  → Final cleanup...")
                Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
            
            adata.obsm['X_lsi'] = Z
            
            print(f"  ✓ LSI matrix cleaned and L2-normalized")
            print(f"  LSI shape: {Z.shape}")
            print(f"  LSI range: [{Z.min():.6f}, {Z.max():.6f}]")
            print(f"  LSI mean: {Z.mean():.6f}, std: {Z.std():.6f}")
            
        except Exception as e:
            print(f"  ✗ LSI failed: {e}")
            print(f"  → Trying with reduced components...")
            
            # Fallback: Try with fewer components
            n_comps_fallback = min(50, adata.n_vars - 1, adata.n_obs - 1)
            print(f"  Retrying with n_comps={n_comps_fallback}...")
            
            svd = TruncatedSVD(
                n_components=n_comps_fallback,
                random_state=0,
                algorithm='arpack',
                n_iter=10
            )
            Z = svd.fit_transform(adata.X)
            Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize
            row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
            row_norms = np.where(row_norms == 0, 1, row_norms)
            Z = Z / row_norms
            Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
            
            adata.obsm['X_lsi'] = Z
            print(f"  ✓ LSI complete with fallback")
        
        print(f"Memory after LSI: {get_memory_usage():.2f} GB")
        
        gc.collect()
        return adata
    
    @staticmethod
    def run_incremental_pca(adata, n_comps=50, batch_size=10000):
        """Incremental PCA for large datasets"""
        X = adata.X
        print(f"  Using IncrementalPCA (batch_size={batch_size})...")
        
        ipca = IncrementalPCA(n_components=n_comps, batch_size=batch_size)
        
        n_batches = int(np.ceil(X.shape[0] / batch_size))
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, X.shape[0])
            batch = X[start:end].toarray() if issparse(X) else X[start:end]
            ipca.partial_fit(batch)
        
        Z = np.zeros((X.shape[0], n_comps))
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, X.shape[0])
            batch = X[start:end].toarray() if issparse(X) else X[start:end]
            Z[start:end] = ipca.transform(batch)
        
        adata.obsm['X_pca'] = Z
        return adata
    
    def batch_correction(self, adata: ad.AnnData) -> ad.AnnData:
        """Harmony batch correction with robust NaN handling"""
        print(f"\n{'='*70}")
        print("STEP 4: HARMONY BATCH CORRECTION")
        print(f"{'='*70}")
        print(f"Memory: {get_memory_usage():.2f} GB")
        
        # Determine input embedding
        embedding_key = 'X_lsi' if self.mode == 'SCATAC' else 'X_pca'
        
        if embedding_key not in adata.obsm:
            raise KeyError(f"Required embedding '{embedding_key}' not found in adata.obsm")
        
        print(f"Using '{embedding_key}' for Harmony...")
        
        # Get data matrix and ensure it's clean
        data_matrix = np.asarray(adata.obsm[embedding_key], dtype=np.float64, order='C')
        
        print(f"  Input shape: {data_matrix.shape}")
        print(f"  Input range: [{data_matrix.min():.6f}, {data_matrix.max():.6f}]")
        
        # CRITICAL: Check for NaN/Inf
        n_nan = np.isnan(data_matrix).sum()
        n_inf = np.isinf(data_matrix).sum()
        
        if n_nan > 0 or n_inf > 0:
            print(f"  ⚠ Found {n_nan} NaN and {n_inf} Inf values")
            print(f"  → Cleaning input data...")
            data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=1e8, neginf=-1e8)
        
        # Check for zero-norm rows
        row_norms = np.linalg.norm(data_matrix, axis=1)
        n_zero_rows = (row_norms == 0).sum()
        
        if n_zero_rows > 0:
            print(f"  ⚠ Found {n_zero_rows} zero-norm rows")
            print(f"  → Adding jitter to zero-norm rows...")
            zero_mask = (row_norms == 0)
            jitter = np.random.RandomState(42).randn(n_zero_rows, data_matrix.shape[1]) * 1e-8
            data_matrix[zero_mask] += jitter
        
        # Final verification
        if np.isnan(data_matrix).any() or np.isinf(data_matrix).any():
            print(f"  ✗ ERROR: Still have NaN/Inf after cleaning!")
            print(f"  → Applying aggressive cleanup...")
            data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure no all-zero rows
            row_sums = np.abs(data_matrix).sum(axis=1)
            if (row_sums == 0).any():
                print(f"  → Replacing all-zero rows with small random values...")
                zero_rows = (row_sums == 0)
                data_matrix[zero_rows] = np.random.RandomState(42).randn(
                    zero_rows.sum(), data_matrix.shape[1]
                ) * 1e-10
        
        print(f"  ✓ Input data verified clean")
        print(f"  Final range: [{data_matrix.min():.6f}, {data_matrix.max():.6f}]")
        
        # Run Harmony
        print(f"  Running Harmony (max_iter={self.harmony_max_iter})...")
        meta_data = adata.obs
        
        try:
            ho = hpy.run_harmony(
                data_matrix,
                meta_data,
                vars_use=[self.batch_key],
                max_iter_harmony=self.harmony_max_iter,
                verbose=False
            )
            
            adata.obsm['X_harmony'] = ho.Z_corr.T
            
            # Verify output
            harmony_matrix = adata.obsm['X_harmony']
            if np.isnan(harmony_matrix).any() or np.isinf(harmony_matrix).any():
                print(f"  ⚠ Harmony output contains NaN/Inf, cleaning...")
                adata.obsm['X_harmony'] = np.nan_to_num(
                    harmony_matrix, nan=0.0, posinf=0.0, neginf=0.0
                )
            
            print(f"  ✓ Harmony complete")
            print(f"  Output shape: {adata.obsm['X_harmony'].shape}")
            
        except Exception as e:
            print(f"  ✗ Harmony failed: {e}")
            print(f"  → Using uncorrected embeddings...")
            # Fallback: Use original embeddings
            adata.obsm['X_harmony'] = data_matrix
        
        print(f"Memory after Harmony: {get_memory_usage():.2f} GB")
        
        return adata
    
    def clustering_and_umap(self, adata: ad.AnnData) -> ad.AnnData:
        """Clustering and UMAP"""
        print(f"\n{'='*70}")
        print("STEP 5: CLUSTERING & UMAP")
        print(f"{'='*70}")
        
        sc.pp.neighbors(adata, use_rep='X_harmony', n_neighbors=30)
        sc.tl.leiden(adata, resolution=self.leiden_resolution)
        sc.tl.umap(adata)
        
        n_clusters = adata.obs['leiden'].nunique()
        print(f"✓ {n_clusters} clusters, UMAP computed")
        
        return adata
    
    def save_results(self, adata: ad.AnnData, output_path: str):
        """Save results"""
        print(f"\n{'='*70}")
        print("STEP 6: SAVING")
        print(f"{'='*70}")
        
        # Final check
        print(f"Final dataset:")
        print(f"  Cells: {adata.n_obs:,}")
        print(f"  Genes (HVG): {adata.n_vars:,}")
        print(f"  Layers: {list(adata.layers.keys())}")
        if 'raw_counts' in adata.layers:
            print(f"    ✓ raw_counts preserved for CNV")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path, compression='gzip')
        
        size_mb = Path(output_path).stat().st_size / (1024**2)
        print(f"✓ Saved: {size_mb:.2f} MB")
    
    def run_integration(self, input_path: str, output_path: str, pattern: str = "*.h5ad"):
        """Run complete pipeline"""
        print(f"\n{'#'*70}")
        print(f"# INTEGRATION PIPELINE - {self.mode}")
        print(f"# Mouse removal: {self.remove_mouse}")
        print(f"# Gene collapse: {self.collapse_genes}")
        print(f"# Cell filtering: NO (keeps G0 cells)")
        print(f"{'#'*70}")
        
        # Load
        adatas = self.load_h5ad_files(input_path, pattern)
        
        # Concatenate
        adata = self.concatenate_data(adatas)
        
        # Gene collapse (with layer preservation!)
        if self.mode == 'SCRNA' and self.collapse_genes:
            adata = collapse_genes_preserve_layers(adata, remove_mouse=self.remove_mouse)
            # Optional: Add genomic annotations
            if self.annotate_genes:
                adata = self.annotate_with_genomic_coords(adata)
        
        # Preprocess
        if self.mode == 'SCRNA':
            adata = self.preprocess_scrna(adata)
        else:
            adata = self.preprocess_scatac(adata)
            
        # Batch correction
        adata = self.batch_correction(adata)
        # Clustering
        adata = self.clustering_and_umap(adata)

        # Save
        self.save_results(adata, output_path)
        print(f"\n{'#'*70}")
        print("# INTEGRATION COMPLETE")
        print(f"{'#'*70}\n")
        return adata


def main():
    parser = argparse.ArgumentParser(description='Optimized single-cell integration')
    
    parser.add_argument('--mode', '-m', required=True, choices=['scRNA', 'scATAC'])
    parser.add_argument('--input', '-i', required=True, help='Input directory')
    parser.add_argument('--output', '-o', required=True, help='Output h5ad file')
    parser.add_argument('--pattern', '-p', default='*.h5ad')
    parser.add_argument('--n-top-genes', type=int, default=2000)
    parser.add_argument('--n-comps', type=int, default=50)
    parser.add_argument('--batch-key', default='Condition')
    parser.add_argument('--leiden-resolution', type=float, default=0.5)
    parser.add_argument('--no-scaling', action='store_true', help='Skip scaling for scRNA')
    parser.add_argument('--no-collapse', action='store_true', help='Skip gene collapse')
    parser.add_argument('--keep-mouse', action='store_true', help='Keep mm10 genes')
    parser.add_argument('--no-annotate', action='store_true', help='Skip genomic annotation')
    parser.add_argument('--plot-dir', default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    integrator = SingleCellIntegrator(
        mode=args.mode,
        n_top_genes=args.n_top_genes,
        n_comps=args.n_comps,
        batch_key=args.batch_key,
        leiden_resolution=args.leiden_resolution,
        apply_scaling=not args.no_scaling,
        collapse_genes=not args.no_collapse,
        remove_mouse=not args.keep_mouse,
        annotate_genes=not args.no_annotate,
        plot_dir=args.plot_dir
    )
    
    try:
        integrator.run_integration(args.input, args.output, args.pattern)
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()