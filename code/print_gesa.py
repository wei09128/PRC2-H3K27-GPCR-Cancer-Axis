#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd

# Usage:
#   python collect_gsea_terms.py "F:\H3K27\Data\CCC_BC\GSEA_Prerank_Results"
#
# If no argument is given, it falls back to this default path:
DEFAULT_FOLDER = "/mnt/f/H3K27/Data/CCC_EC/GSEA_Prerank_Results"

def get_cell_type_from_name(fname: str) -> str:
    ct = fname
    for suf in [
        "_GSEA_Prerank_Results.csv",
        "_GSEA_Results.csv",
        "_GSEA_Prerank.csv",
        "_GSEA.csv",
        ".csv",
    ]:
        if ct.endswith(suf):
            ct = ct[: -len(suf)]
            break
    return ct

def main(folder_path: str | None = None):
    folder = Path(folder_path or DEFAULT_FOLDER)

    if not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    rows = []

    for csv_path in sorted(folder.glob("*.csv")):
        cell_type = get_cell_type_from_name(csv_path.name)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Could not read {csv_path}: {e}", file=sys.stderr)
            continue

        # Make sure required columns exist
        if not {"Term", "NES"}.issubset(df.columns):
            print(f"Skipping {csv_path} (missing Term or NES column)", file=sys.stderr)
            continue

        # Optional: keep only significant pathways (FDR q-val < 0.25)
        fdr_col = None
        for cand in ["FDR q-val", "FDR_qval", "FDR_q-value"]:
            if cand in df.columns:
                fdr_col = cand
                break

        if fdr_col is not None:
            df = df[df[fdr_col] < 0.25].copy()

        if df.empty:
            continue

        # Collect rows
        for _, r in df.iterrows():
            rows.append(
                {
                    "Cell_Type": cell_type,
                    "Term": r["Term"],
                    "NES": r["NES"],
                }
            )

    if not rows:
        print("No rows collected (maybe all FDR >= 0.25?)", file=sys.stderr)
        return

    out = pd.DataFrame(rows)
    # sort by cell type, then by |NES|
    out["absNES"] = out["NES"].abs()
    out = out.sort_values(["Cell_Type", "absNES"], ascending=[True, False])
    out = out[["Cell_Type", "Term", "NES"]]

    # Print as CSV to stdout
    print(out.to_csv(index=False))

if __name__ == "__main__":
    folder_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(folder_arg)
