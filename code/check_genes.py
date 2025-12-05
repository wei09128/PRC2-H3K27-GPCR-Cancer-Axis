#!/usr/bin/env python3
"""
Check cell phenotypes from epigenetic enrichment data
"""
import pandas as pd
import sys
from typing import Dict, List, Tuple

# ============================================================================
# PERICYTE MARKERS
# ============================================================================
LEAKY_MARKERS = {
    'destabilization': ['ANGPT2', 'VEGFA', 'VEGFC'],
    'ECM_degradation': ['MMP2', 'MMP9', 'MMP14', 'PLAU'],
    'permeability': ['VEGFA', 'NOS3', 'PTGS2'],
}

STABLE_MARKERS = {
    'coverage': ['PDGFRB', 'CSPG4', 'RGS5'],
    'contractile': ['ACTA2', 'TAGLN', 'MYH11', 'CNN1'],
    'maturation': ['RGS5', 'ABCC9', 'KCNJ8'],
    'basement_membrane': ['COL4A1', 'COL4A2', 'LAMA2', 'LAMA4'],
}

# ============================================================================
# MACROPHAGE MARKERS
# ============================================================================
# M1-like: Pro-inflammatory, anti-tumor
M1_MARKERS = {
    'pro_inflammatory': ['TNF', 'IL1B', 'IL6', 'IL12A', 'IL12B'],
    'antigen_presentation': ['CD80', 'CD86', 'HLA-DRA', 'HLA-DRB1'],
    'ROS_production': ['NOS2', 'CYBB'],
    'cytotoxic': ['TNFSF10', 'FAS', 'CASP1'],
}

# M2-like: Immunosuppressive, pro-tumor
M2_MARKERS = {
    'immunosuppression': ['IL10', 'TGFB1', 'CD163', 'CD206', 'MRC1'],
    'tissue_remodeling': ['MMP9', 'MMP2', 'MMP12', 'VEGFA'],
    'alternative_activation': ['ARG1', 'ARG2', 'ALOX15', 'IL4R'],
    'angiogenesis': ['VEGFA', 'VEGFC', 'PDGFB', 'FGF2'],
}

# General macrophage markers
MACRO_GENERAL = {
    'core_identity': ['CD68', 'CD14', 'ITGAM', 'CSF1R', 'FCGR1A'],
    'monocyte_derived': ['S100A8', 'S100A9', 'S100A12', 'LYZ'],
}

# ============================================================================
# MONOCYTE MARKERS
# ============================================================================
# Classical monocytes (inflammatory)
CLASSICAL_MONOCYTE = {
    'classical': ['CD14', 'FCGR3A', 'CCR2', 'CD16', 'S100A8', 'S100A9'],
    'inflammatory': ['IL1B', 'TNF', 'CCL2', 'CCL3', 'CXCL8'],
}

# Non-classical monocytes (patrolling, pro-angiogenic)
NONCLASSICAL_MONOCYTE = {
    'nonclassical': ['FCGR3A', 'CD16', 'CX3CR1', 'HLA-DR'],
    'patrolling': ['CDKN1C', 'LILRB2'],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_matching_columns(df: pd.DataFrame, keywords: List[str]) -> List[str]:
    """Find columns matching any of the keywords"""
    matched = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw.lower() in col_lower for kw in keywords):
            matched.append(col)
    return matched

def check_markers(genes: List[str], marker_dict: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Check which markers from marker_dict are present in genes"""
    found = []
    genes_upper = [g.upper() for g in genes]
    for category, markers in marker_dict.items():
        for marker in markers:
            if marker.upper() in genes_upper:
                found.append((category, marker))
    return found

def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ============================================================================
# CELL TYPE CHECKERS
# ============================================================================

def check_pericytes(df: pd.DataFrame):
    """Check pericyte phenotype"""
    print_section("PERICYTE ANALYSIS")
    
    peri_cols = find_matching_columns(df, ['pericyte'])
    if not peri_cols:
        print("❌ No pericyte columns found")
        return
    
    for col in peri_cols:
        print(f"\n📊 Column: {col}")
        genes = df[col].dropna().astype(str).tolist()
        genes = [g for g in genes if g.upper() not in ['NAN', 'NONE', '']]
        print(f"   Total genes: {len(genes)}")
        
        found_leaky = check_markers(genes, LEAKY_MARKERS)
        found_stable = check_markers(genes, STABLE_MARKERS)
        
        print("\n   🔴 Leaky/Destabilization markers:")
        if found_leaky:
            for cat, gene in found_leaky:
                print(f"      ✓ {gene} ({cat})")
        else:
            print("      ✗ None found")
        
        print("\n   🟢 Stable/Mature markers:")
        if found_stable:
            for cat, gene in found_stable:
                print(f"      ✓ {gene} ({cat})")
        else:
            print("      ✗ None found")
        
        # Verdict
        n_leaky, n_stable = len(found_leaky), len(found_stable)
        print(f"\n   📊 Score: {n_stable} stable vs {n_leaky} leaky")
        
        if n_stable > n_leaky * 2:
            print("   ✅ STABLE/MATURE phenotype")
        elif n_leaky > n_stable * 2:
            print("   ⚠️  LEAKY/DESTABILIZED phenotype")
        else:
            print("   ⚖️  MIXED phenotype")

def check_macrophages(df: pd.DataFrame):
    """Check macrophage polarization"""
    print_section("MACROPHAGE ANALYSIS")
    
    macro_cols = find_matching_columns(df, ['macrophage', 'macro'])
    if not macro_cols:
        print("❌ No macrophage columns found")
        return
    
    for col in macro_cols:
        print(f"\n📊 Column: {col}")
        genes = df[col].dropna().astype(str).tolist()
        genes = [g for g in genes if g.upper() not in ['NAN', 'NONE', '']]
        print(f"   Total genes: {len(genes)}")
        
        found_general = check_markers(genes, MACRO_GENERAL)
        found_m1 = check_markers(genes, M1_MARKERS)
        found_m2 = check_markers(genes, M2_MARKERS)
        
        print("\n   🔵 General macrophage markers:")
        if found_general:
            for cat, gene in found_general:
                print(f"      ✓ {gene} ({cat})")
        else:
            print("      ✗ None found")
        
        print("\n   🔴 M1-like (anti-tumor) markers:")
        if found_m1:
            for cat, gene in found_m1:
                print(f"      ✓ {gene} ({cat})")
        else:
            print("      ✗ None found")
        
        print("\n   🟢 M2-like (pro-tumor) markers:")
        if found_m2:
            for cat, gene in found_m2:
                print(f"      ✓ {gene} ({cat})")
        else:
            print("      ✗ None found")
        
        # Verdict
        n_m1, n_m2 = len(found_m1), len(found_m2)
        print(f"\n   📊 Score: {n_m1} M1-like vs {n_m2} M2-like")
        
        if n_m1 > n_m2 * 2:
            print("   ✅ M1-like (ANTI-TUMOR) polarization")
            print("   💡 Interpretation: Pro-inflammatory, potentially anti-tumor")
        elif n_m2 > n_m1 * 2:
            print("   ⚠️  M2-like (PRO-TUMOR) polarization")
            print("   💡 Interpretation: Immunosuppressive, tumor-supportive")
        else:
            print("   ⚖️  MIXED or INTERMEDIATE polarization")

def check_monocytes(df: pd.DataFrame):
    """Check monocyte phenotype"""
    print_section("MONOCYTE ANALYSIS")
    
    mono_cols = find_matching_columns(df, ['monocyte', 'mono'])
    if not mono_cols:
        print("❌ No monocyte columns found")
        return
    
    for col in mono_cols:
        print(f"\n📊 Column: {col}")
        genes = df[col].dropna().astype(str).tolist()
        genes = [g for g in genes if g.upper() not in ['NAN', 'NONE', '']]
        print(f"   Total genes: {len(genes)}")
        
        found_classical = check_markers(genes, CLASSICAL_MONOCYTE)
        found_nonclassical = check_markers(genes, NONCLASSICAL_MONOCYTE)
        
        print("\n   🔴 Classical (inflammatory) markers:")
        if found_classical:
            for cat, gene in found_classical:
                print(f"      ✓ {gene} ({cat})")
        else:
            print("      ✗ None found")
        
        print("\n   🔵 Non-classical (patrolling) markers:")
        if found_nonclassical:
            for cat, gene in found_nonclassical:
                print(f"      ✓ {gene} ({cat})")
        else:
            print("      ✗ None found")
        
        # Verdict
        n_class, n_nonclass = len(found_classical), len(found_nonclassical)
        print(f"\n   📊 Score: {n_class} classical vs {n_nonclass} non-classical")
        
        if n_class > n_nonclass * 2:
            print("   ✅ CLASSICAL (inflammatory) phenotype")
            print("   💡 Interpretation: Pro-inflammatory recruitment")
        elif n_nonclass > n_class * 2:
            print("   ⚠️  NON-CLASSICAL (patrolling) phenotype")
            print("   💡 Interpretation: Tissue surveillance, potentially pro-angiogenic")
        else:
            print("   ⚖️  MIXED phenotype")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_all_cells(csv_file: str):
    """Run all cell type analyses"""
    print("=" * 70)
    print("CELL PHENOTYPE ANALYZER - H3K27me3 Enrichment")
    print("=" * 70)
    print(f"\n📁 File: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"📋 Columns found: {list(df.columns)}")
    
    # Run all analyses
    check_macrophages(df)
    check_monocytes(df)
    check_pericytes(df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n💡 Note: H3K27me3 marks REPRESSED chromatin")
    print("   Genes found here are likely SILENCED in these cells")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: python check_cell_phenotypes.py <enrichment_csv>")
        print("\nThis script checks:")
        print("  • Pericytes: leaky vs stable vascular phenotype")
        print("  • Macrophages: M1 vs M2 polarization")
        print("  • Monocytes: classical vs non-classical")
        sys.exit(1)
    
    analyze_all_cells(sys.argv[1])