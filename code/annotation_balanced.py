#!/usr/bin/env python3
"""
Balanced Multi-Evidence Cell Type Annotation Pipeline
Integrates: Sample metadata (prior) + Markers (evidence) + UMAP clusters (consistency)
"""
import scanpy as sc
import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
sc.settings.verbosity = 2


class BalancedCellTypeAnnotator:
    """
    Multi-evidence annotation strategy:
    1. Sample metadata gives cancer type prior for cancer cells
    2. Marker scores provide primary evidence
    3. UMAP clusters enforce spatial consistency
    """
    
    def __init__(self, mode: str = 'scRNA', use_cnv: bool = False, plot_dir: Optional[str] = None):
        self.mode = mode.upper()
        self.use_cnv = use_cnv
        self.plot_dir = plot_dir
        self.markers = {}
        self.sample_cancer_map = {}
        
    def load_sample_metadata(self, metadata_dict: Dict[str, str]):
        """Load sample -> cancer type mapping"""
        print(f"\nLoading sample metadata:")
        self.sample_cancer_map = metadata_dict
        
        cancer_counts = Counter(metadata_dict.values())
        print(f"  ✓ Loaded {len(metadata_dict)} samples")
        for cancer_type, count in cancer_counts.most_common():
            print(f"    {cancer_type}: {count} samples")
        
        return self.sample_cancer_map
    
    def load_markers(self, marker_file: str):
        """Load marker definitions from YAML"""
        print(f"\nLoading markers from: {marker_file}")
        
        with open(marker_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.markers = config.get('cell_types', {})
        
        print(f"✓ Loaded {len(self.markers)} cell types")
        for name, info in sorted(self.markers.items()):
            n_markers = len(info.get('markers', []))
            n_neg = len(info.get('negative_markers', []))
            print(f"  {name}: {n_markers} positive, {n_neg} negative markers")
        
        return self.markers
    
    def score_markers(self, adata, use_raw: bool = True):
        """Score markers WITHOUT copying data - memory efficient"""
        print(f"\n{'='*70}")
        print("SCORING MARKER GENES")
        print(f"{'='*70}")
        
        # Determine data source
        if use_raw:
            if 'raw_counts' in adata.layers:
                print("  Data source: adata.layers['raw_counts']")
                score_data = adata
                use_layer = 'raw_counts'
                use_raw_param = False
            elif adata.raw is not None:
                print("  Data source: adata.raw")
                score_data = adata
                use_layer = None
                use_raw_param = True
            else:
                print("  Data source: ⚠ WARNING: Using adata.X (no raw data found)")
                score_data = adata
                use_layer = None
                use_raw_param = False
        else:
            print("  Data source: adata.X (normalized)")
            score_data = adata
            use_layer = None
            use_raw_param = False
        
        # Get available genes
        if use_layer:
            available_genes = set(score_data.var_names)
        elif use_raw_param and adata.raw is not None:
            available_genes = set(adata.raw.var_names)
        else:
            available_genes = set(score_data.var_names)
        
        # Score ALL cell types equally
        for cell_type, info in self.markers.items():
            markers = info.get('markers', [])
            available = [m for m in markers if m in available_genes]
            
            if available:
                score_name = f'{cell_type}_score'
                
                if use_layer:
                    sc.tl.score_genes(
                        score_data, 
                        gene_list=available, 
                        score_name=score_name, 
                        use_raw=False,
                        layer=use_layer
                    )
                else:
                    sc.tl.score_genes(
                        score_data, 
                        gene_list=available, 
                        score_name=score_name, 
                        use_raw=use_raw_param
                    )
                
                print(f"  ✓ {cell_type}: {len(available)}/{len(markers)} markers")
            else:
                print(f"  ✗ {cell_type}: No markers found")
                adata.obs[f'{cell_type}_score'] = 0.0
            
            # Score negative markers (penalize presence)
            neg_markers = info.get('negative_markers', [])
            if neg_markers:
                neg_available = [m for m in neg_markers if m in available_genes]
                if neg_available:
                    neg_score_name = f'{cell_type}_neg_score'
                    
                    if use_layer:
                        sc.tl.score_genes(
                            score_data,
                            gene_list=neg_available,
                            score_name=neg_score_name,
                            use_raw=False,
                            layer=use_layer
                        )
                    else:
                        sc.tl.score_genes(
                            score_data,
                            gene_list=neg_available,
                            score_name=neg_score_name,
                            use_raw=use_raw_param
                        )
                    
                    # Subtract negative score from positive score
                    adata.obs[f'{cell_type}_score'] -= adata.obs[neg_score_name]
                    print(f"    - Penalized by {len(neg_available)} negative markers")
        
        return adata
    
    def annotate_balanced(self, adata, sample_col: str = 'Sample_ID', 
                          leiden_col: str = 'leiden',
                          marker_weight: float = 0.5,
                          sample_weight: float = 0.5,
                          cluster_weight: float = 0.0):
        """
        Balanced annotation combining three evidence sources:
        
        CRITICAL: Sample metadata is ABSOLUTE TRUTH for cancer type distinction
        - BC sample → Can only be BC or non-cancer (NEVER OC/EC/GC)
        - OC sample → Can only be OC or non-cancer (NEVER BC/EC/GC)
        - EC sample → Can only be EC or non-cancer (NEVER BC/OC/GC)
        - GC sample → Can only be GC or non-cancer (NEVER BC/OC/EC)
        
        1. Marker scores: Primary evidence from gene expression
        2. Sample metadata: BLOCKS incorrect cancer types completely
        3. Cluster consistency: Spatial smoothing (optional)
        """
        print(f"\n{'='*70}")
        print("BALANCED MULTI-EVIDENCE ANNOTATION")
        print(f"  Marker weight: {marker_weight:.1%}")
        print(f"  Sample weight: {sample_weight:.1%}")
        print(f"  Cluster weight: {cluster_weight:.1%}")
        print(f"{'='*70}")
        
        n_cells = len(adata)
        
        # Get all score columns
        score_columns = [col for col in adata.obs.columns 
                        if col.endswith('_score') and not col.endswith('_neg_score')]
        score_columns = [col for col in score_columns 
                        if col.replace('_score', '') in self.markers]
        
        if not score_columns:
            print("  ✗ No score columns found!")
            adata.obs['cell_type'] = 'Unassigned'
            return adata
        
        cell_type_names = [col.replace('_score', '') for col in score_columns]
        cancer_types = ['BC', 'OC', 'EC', 'GC']  # Known cancer types
        
        # ==== STEP 1: Marker Evidence ====
        print(f"\n1. Computing marker evidence...")
        score_matrix = adata.obs[score_columns].values
        
        # Apply thresholds
        threshold_matrix = np.zeros_like(score_matrix, dtype=bool)
        for i, cell_type in enumerate(cell_type_names):
            threshold = self.markers[cell_type].get('marker_threshold', 0.5)
            threshold_matrix[:, i] = score_matrix[:, i] >= threshold
        
        # Normalize scores to [0, 1] range per cell (softmax-like)
        marker_evidence = score_matrix.copy()
        marker_evidence[~threshold_matrix] = -np.inf
        
        # Convert to probabilities
        marker_probs = np.exp(marker_evidence - marker_evidence.max(axis=1, keepdims=True))
        marker_probs = marker_probs / (marker_probs.sum(axis=1, keepdims=True) + 1e-10)
        marker_probs[np.isinf(marker_evidence).all(axis=1)] = 0  # Unassigned
        
        # ==== STEP 2: Sample Cancer Type Prior (ABSOLUTE TRUTH) ====
        print(f"2. Applying sample metadata as ABSOLUTE constraint...")
        sample_probs = np.ones_like(score_matrix)  # Start with all allowed
        
        if sample_col in adata.obs.columns:
            for i, cell_sample in enumerate(adata.obs[sample_col]):
                if cell_sample in self.sample_cancer_map:
                    sample_cancer = self.sample_cancer_map[cell_sample]
                    
                    if sample_cancer in cancer_types:
                        # ABSOLUTE RULE: Block ALL other cancer types
                        for other_cancer in cancer_types:
                            if other_cancer != sample_cancer and other_cancer in cell_type_names:
                                other_idx = cell_type_names.index(other_cancer)
                                sample_probs[i, other_idx] = 0.0  # BLOCKED
                        
                        # Strong boost to matching cancer type
                        if sample_cancer in cell_type_names:
                            cancer_idx = cell_type_names.index(sample_cancer)
                            sample_probs[i, cancer_idx] = 100.0  # Very strong
                        
                        # Allow all non-cancer types
                        for j, ct in enumerate(cell_type_names):
                            if ct not in cancer_types and sample_probs[i, j] > 0:
                                sample_probs[i, j] = 1.0
                                
                    elif sample_cancer == 'Normal Breast Cells':
                        # BLOCK ALL cancer types
                        for cancer in cancer_types:
                            if cancer in cell_type_names:
                                c_idx = cell_type_names.index(cancer)
                                sample_probs[i, c_idx] = 0.0  # BLOCKED
                        
                        # Allow all non-cancer types equally
                        for j, ct in enumerate(cell_type_names):
                            if ct not in cancer_types:
                                sample_probs[i, j] = 1.0
            
            # Count blocked assignments
            n_blocked = (sample_probs == 0.0).sum()
            print(f"  ✓ Blocked {n_blocked:,} impossible cancer type assignments")
            
            # Normalize (zeros stay zero)
            row_sums = sample_probs.sum(axis=1, keepdims=True)
            sample_probs = np.where(row_sums > 0, 
                                   sample_probs / row_sums, 
                                   1.0 / len(cell_type_names))
        else:
            print("  ⚠ Sample column not found, skipping sample prior")
            sample_probs = np.ones_like(score_matrix) / len(cell_type_names)
        
        # ==== STEP 3: Cluster Consistency ====
        print(f"3. Computing cluster consistency...")
        cluster_probs = np.zeros_like(score_matrix)
        
        if leiden_col in adata.obs.columns:
            # For each cluster, compute majority cell type from markers
            clusters = adata.obs[leiden_col].values
            
            # Get best marker-based assignment for each cell
            best_marker_types = np.argmax(marker_probs, axis=1)
            
            for cluster_id in np.unique(clusters):
                cluster_mask = (clusters == cluster_id)
                
                # Majority vote within cluster
                cluster_assignments = best_marker_types[cluster_mask]
                cluster_assignments = cluster_assignments[marker_probs[cluster_mask].max(axis=1) > 0]
                
                if len(cluster_assignments) > 0:
                    # Count cell types in this cluster
                    type_counts = np.bincount(cluster_assignments, 
                                            minlength=len(cell_type_names))
                    cluster_probs[cluster_mask] = type_counts / (type_counts.sum() + 1e-10)
                else:
                    # No strong assignments - uniform
                    cluster_probs[cluster_mask] = 1.0 / len(cell_type_names)
        else:
            print("  ⚠ Leiden column not found, skipping cluster consistency")
            cluster_probs = np.ones_like(score_matrix) / len(cell_type_names)
        
        # ==== STEP 4: Combine Evidence with HARD constraints ====
        print(f"\n4. Combining evidence (sample constraints are absolute)...")
        
        # First apply sample constraints (zeros stay zeros!)
        combined_probs = marker_probs * sample_probs
        
        # If cluster weight > 0, add cluster consistency
        if cluster_weight > 0 and leiden_col in adata.obs.columns:
            combined_probs = (marker_weight * marker_probs * sample_probs + 
                             cluster_weight * cluster_probs * sample_probs)
        
        # Normalize
        row_sums = combined_probs.sum(axis=1, keepdims=True)
        combined_probs = np.where(row_sums > 0,
                                 combined_probs / row_sums,
                                 0)  # If all zeros, stays unassigned
        
        # Assign cell types
        best_indices = np.argmax(combined_probs, axis=1)
        best_probs = np.max(combined_probs, axis=1)
        
        # Minimum confidence threshold
        min_confidence = 0.3
        
        adata.obs['cell_type'] = [
            cell_type_names[idx] if prob >= min_confidence else 'Unassigned'
            for idx, prob in zip(best_indices, best_probs)
        ]
        adata.obs['cell_type_confidence'] = best_probs
        
        # Store individual evidence scores
        adata.obs['marker_evidence'] = marker_probs.max(axis=1)
        adata.obs['sample_evidence'] = sample_probs.max(axis=1)
        adata.obs['cluster_evidence'] = cluster_probs.max(axis=1)
        
        # Add assignment reason
        adata.obs['assignment_reason'] = ''
        for cell_type in cell_type_names:
            mask = (adata.obs['cell_type'] == cell_type)
            if mask.any():
                adata.obs.loc[mask, 'assignment_reason'] = 'balanced_multi_evidence'
        
        # ==== Summary ====
        print(f"\n{'='*70}")
        print("ANNOTATION RESULTS")
        print(f"{'='*70}")
        
        print(f"\nCell type distribution:")
        for ct, count in adata.obs['cell_type'].value_counts().head(20).items():
            pct = 100 * count / n_cells
            mean_conf = adata.obs[adata.obs['cell_type'] == ct]['cell_type_confidence'].mean()
            mean_marker = adata.obs[adata.obs['cell_type'] == ct]['marker_evidence'].mean()
            print(f"  {ct}: {count:,} ({pct:.1f}%)")
            print(f"    → Confidence: {mean_conf:.3f}, Marker: {mean_marker:.3f}")
        
        # Sample-level summary with validation
        if sample_col in adata.obs.columns:
            print(f"\nSample-level validation (checking cancer type constraints):")
            print(f"{'='*70}")
            
            violations = []
            
            for sample in sorted(adata.obs[sample_col].unique()):
                sample_mask = adata.obs[sample_col] == sample
                expected_cancer = self.sample_cancer_map.get(sample, 'Unknown')
                
                # Count cancer types in this sample
                cancer_counts = {}
                for cancer in cancer_types:
                    count = (adata.obs[sample_mask]['cell_type'] == cancer).sum()
                    if count > 0:
                        cancer_counts[cancer] = count
                
                # Check for violations
                if expected_cancer in cancer_types:
                    # Should ONLY have the expected cancer type
                    wrong_cancers = [c for c in cancer_counts.keys() if c != expected_cancer]
                    if wrong_cancers:
                        violations.append(f"  ✗ {sample}: Expected {expected_cancer} only, but found {wrong_cancers}")
                elif expected_cancer == 'Normal Breast Cells':
                    # Should have NO cancer types
                    if cancer_counts:
                        violations.append(f"  ✗ {sample}: Expected no cancer, but found {list(cancer_counts.keys())}")
                
                # Show summary for first 10 samples
                if len([s for s in sorted(adata.obs[sample_col].unique()) if s <= sample]) <= 10:
                    top_types = adata.obs[sample_mask]['cell_type'].value_counts().head(3)
                    print(f"\n  {sample} (expected: {expected_cancer})")
                    for ct, count in top_types.items():
                        pct = 100 * count / sample_mask.sum()
                        is_cancer = ct in cancer_types
                        marker = "🔴" if is_cancer else "🔵"
                        print(f"    {marker} {ct}: {count} ({pct:.1f}%)")
            
            if violations:
                print(f"\n{'='*70}")
                print(f"⚠️  CONSTRAINT VIOLATIONS DETECTED:")
                print(f"{'='*70}")
                for v in violations:
                    print(v)
                print(f"\n⚠️  These should be ZERO with absolute constraints!")
            else:
                print(f"\n{'='*70}")
                print(f"✓ NO VIOLATIONS: All cancer type constraints respected!")
                print(f"{'='*70}")
        
        return adata

    def run(self, input_path: str, output_path: str, marker_file: str, 
            sample_metadata: Dict[str, str], use_raw: bool = True,
            sample_col: str = 'Sample_ID', leiden_col: str = 'leiden'):
        """Main balanced annotation pipeline"""
        print(f"\n{'#'*70}")
        print(f"BALANCED MULTI-EVIDENCE CELL TYPE ANNOTATION")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"{'#'*70}")

        # Load data
        print("\n1. Loading data...")
        adata = sc.read_h5ad(input_path)
        
        # Filter doublets
        if 'predicted_doublet' in adata.obs.columns:
            adata = adata[adata.obs['predicted_doublet'] == False].copy()
            print(f"  ✓ Filtered doublets")
        
        print(f"  ✓ {adata.n_obs:,} cells, {adata.n_vars:,} genes")
        print(f"  Layers: {list(adata.layers.keys())}")
        print(f"  Has .raw: {adata.raw is not None}")

        # Load metadata and markers
        self.load_sample_metadata(sample_metadata)
        self.load_markers(marker_file)

        # Score markers
        adata = self.score_markers(adata, use_raw=use_raw)
        
        # Balanced annotation
        adata = self.annotate_balanced(
            adata, 
            sample_col=sample_col,
            leiden_col=leiden_col,
            marker_weight=0.5,  # 50% marker evidence
            sample_weight=0.5,  # 50% sample constraint (blocks wrong cancers)
            cluster_weight=0.0  # Cluster consistency disabled
        )
        
        # Save
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        adata.write(output_path)
        print(f"✓ Saved to: {output_path}")
        
        print(f"\n{'#'*70}")
        print("✓ BALANCED ANNOTATION COMPLETE")
        print(f"{'#'*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Balanced multi-evidence cell type annotation'
    )
    parser.add_argument('--input', '-i', required=True, help='Input h5ad')
    parser.add_argument('--output', '-o', required=True, help='Output h5ad')
    parser.add_argument('--mode', '-m', required=True, choices=['scRNA', 'scATAC'])
    parser.add_argument('--markers', required=True, help='Marker YAML')
    parser.add_argument('--use-raw', action='store_true', default=True)
    parser.add_argument('--sample-col', default='Sample_ID', help='Sample ID column')
    parser.add_argument('--leiden-col', default='leiden', help='Leiden cluster column')
    
    args = parser.parse_args()
    
    # Sample metadata mapping (from your data)
    sample_metadata = {
        'A1': 'EC', 'A2': 'EC', 'A3': 'EC', 'A4': 'EC', 'A5': 'EC', 'A6': 'EC',
        'A7': 'OC', 'A8': 'OC', 'A9': 'OC', 'A10': 'OC', 'A11': 'GC',
        'B1': 'Normal Breast Cells', 'B2': 'Normal Breast Cells', 
        'B3': 'Normal Breast Cells', 'B4': 'Normal Breast Cells',
        'B5': 'BC', 'B6': 'BC', 'B7': 'BC', 'B8': 'BC', 'B9': 'BC', 'B10': 'BC',
        'B11': 'BC', 'B12': 'BC', 'B13': 'BC', 'B14': 'BC', 'B14B': 'BC', 'B15': 'BC',
        'C1-1': 'BC', 'C1-2': 'BC', 'C1-3': 'BC', 'C1-4': 'BC', 'C1-5': 'BC',
        'C2-1': 'BC', 'C2-2': 'BC', 'C3-1': 'BC', 'C3-2': 'BC',
        'HCC1143': 'BC', 'MCF7': 'BC', 'SUM149PT': 'BC', 'T47D': 'BC',
        'MM468-5FU1-D214': 'BC', 'MM468-5FU1-D33': 'BC',
        'MM468-5FU2-D171': 'BC', 'MM468-5FU2-D67': 'BC',
        'MM468-5FU3-D202': 'BC', 'MM468-5FU3-D50': 'BC', 'MM468-5FU3-D77': 'BC',
        'MM468-5FUUnt-D33-H3K4me3': 'BC', 'MM468-Untreated-D0-H3K4me3': 'BC',
        'MM468-Untreated-D33-H3K4me3': 'BC'
    }
    
    annotator = BalancedCellTypeAnnotator(mode=args.mode, use_cnv=False)
    
    try:
        annotator.run(
            input_path=args.input,
            output_path=args.output,
            marker_file=args.markers,
            sample_metadata=sample_metadata,
            use_raw=args.use_raw,
            sample_col=args.sample_col,
            leiden_col=args.leiden_col
        )
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())