# summarize_pseudobulk.py
import os, glob, math, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ======== CONFIG ========
# IN_DIR  = "/mnt/f/H3K27/Data/CCC_pseudo/"  # folder with pseudobulk_*_edgeR_DGE.csv
# IN_DIR  = "/mnt/f/H3K27/Data/scATAC_pseudo/"  # folder with pseudobulk_*_edgeR_DGE.csv
# FDR_CUTOFF = 0.05
# LFC_CUTOFF = 0.50

IN_DIR  = "/mnt/f/H3K27/Data/scChi_pseudo/"  # folder with pseudobulk_*_edgeR_DGE.csv
FDR_CUTOFF = 0.05
LFC_CUTOFF = 0.50
OUT_DIR = os.path.join(IN_DIR, "summary")

N_TOP_GENES = 25   # for the cross-CT heatmap
MAX_VOLCANOS = 12  # make grids if you have many CTs
# ========================

os.makedirs(OUT_DIR, exist_ok=True)

def parse_cell_type(path):
    base = os.path.basename(path)
    # pseudobulk_<CT>_edgeR_DGE.csv
    ct = base.replace("pseudobulk_", "").replace("_edgeR_DGE.csv","")
    return ct

# 1) Collate all DGE tables
tables = []
for p in sorted(glob.glob(os.path.join(IN_DIR, "*_edgeR_DGE.csv"))):
    ct = parse_cell_type(p)
    df = pd.read_csv(p)

    # expected columns: gene, logFC, FDR (edgeR)
    needed = {"gene", "logFC", "FDR"}
    if not needed.issubset(df.columns):
        print(f"Skipping {p} (missing columns). Found: {df.columns.tolist()}")
        continue

    df["cell_type"] = ct
    df["neglog10FDR"] = -np.log10(df["FDR"].replace(0, np.nextafter(0, 1)))
    df["is_sig"] = (df["FDR"] < FDR_CUTOFF) & (df["logFC"].abs() > LFC_CUTOFF)
    tables.append(df)

if not tables:
    print("No DGE CSVs found or none had the required columns.")
    sys.exit(0)

# Combine all cell types
all_res = pd.concat(tables, ignore_index=True)

# Work only with significant hits
sig_res = all_res[all_res["is_sig"]].copy()

if sig_res.empty:
    print(f"\n⚠️ No genes pass FDR < {FDR_CUTOFF} and |logFC| > {LFC_CUTOFF}.")
    print("   Skipping bar/dot/heatmap plots that require significant hits.\n")

    # OPTIONAL FALLBACK: uncomment if you still want “top by FDR” even if not formally significant
    # sig_res = (all_res.sort_values("FDR")
    #                   .groupby("cell_type", group_keys=False)
    #                   .head(NTOP))


all_dge = pd.concat(tables, ignore_index=True)
all_dge.to_csv(os.path.join(OUT_DIR, "pseudobulk_DGE_master.csv"), index=False)

# Per-CT summary counts
summary = (all_dge
           .assign(direction=np.where(all_dge["logFC"]>0,"Up","Down"))
           .query("FDR < @FDR_CUTOFF and abs(logFC) > @LFC_CUTOFF")
           .groupby(["cell_type","direction"])
           .size().unstack(fill_value=0).reset_index())
summary["Total"] = summary.get("Up",0)+summary.get("Down",0)
summary.sort_values(["Total","cell_type"], ascending=[False,True], inplace=True)
summary.to_csv(os.path.join(OUT_DIR, "pseudobulk_DGE_sig_counts_by_celltype.csv"), index=False)
print(summary.head(10))

# 2a) Volcano plots per CT (PDF grid batches)
cts = sorted(all_dge["cell_type"].unique())
def volcano(df, ct, ax):
    x = df["logFC"].values
    y = df["neglog10FDR"].values
    sig = (df["FDR"]<FDR_CUTOFF) & (df["logFC"].abs()>LFC_CUTOFF)
    ax.scatter(x[~sig], y[~sig], s=4, alpha=0.4)
    ax.scatter(x[sig],  y[sig],  s=6, alpha=0.8)
    ax.axvline(LFC_CUTOFF,  ls="--", lw=0.7)
    ax.axvline(-LFC_CUTOFF, ls="--", lw=0.7)
    ax.axhline(-math.log10(FDR_CUTOFF), ls="--", lw=0.7)
    ax.set_title(ct, fontsize=9)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 FDR")

for i in range(0, len(cts), MAX_VOLCANOS):
    batch = cts[i:i+MAX_VOLCANOS]
    n = len(batch)
    cols = 4
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    axes = np.array(axes).reshape(-1)
    for j, ct in enumerate(batch):
        df = all_dge[all_dge["cell_type"]==ct]
        volcano(df, ct, axes[j])
    for k in range(j+1, rows*cols):
        axes[k].axis("off")
    fig.suptitle("Pseudobulk DGE Volcano Plots", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"volcanos_batch_{i//MAX_VOLCANOS+1}.png"))
    plt.close(fig)

# 2b) Cross-CT heatmap of top genes (signed logFC)

# if no significant hits at all, just skip this section
if sig_res.empty:
    print("No significant genes (sig_res is empty); skipping heatmap / dotplot.")
else:
    # pick union of top N by FDR among significant genes per CT
    tops = []
    for ct, sub in sig_res.groupby("cell_type"):
        top_ct = sub.sort_values("FDR").head(N_TOP_GENES)
        if top_ct.empty:
            continue
        tops.append(top_ct[["gene", "cell_type"]])

    if not tops:
        print("No top genes to plot (no rows in sig_res after per-CT selection).")
        top_union = []
    else:
        top_union = (
            pd.concat(tops, ignore_index=True)
              .drop_duplicates("gene")["gene"]
              .tolist()
        )

    if not tops or not top_union:
        print("No genes to include in heatmap / dotplot; skipping this step.")
    else:
        # build matrix of logFC for union of top genes
        mat = (all_dge[all_dge["gene"].isin(top_union)]
               .pivot_table(index="gene",
                            columns="cell_type",
                            values="logFC",
                            aggfunc="mean")
               .fillna(0.0))

        if mat.size == 0 or mat.shape[0] == 0 or mat.shape[1] == 0:
            print("Heatmap matrix is empty; skipping heatmap / dotplot.")
        else:
            # ---- column order: use the DGE summary (most DE-rich CTs first) ----
            sig_cts = summary["cell_type"].tolist() if not summary.empty else []
            all_cts = list(mat.columns)

            ct_order = sig_cts + [ct for ct in all_cts if ct not in sig_cts]
            mat = mat.reindex(columns=ct_order)

            # order genes by maximum absolute effect
            order_genes = mat.reindex(
                mat.abs().max(axis=1).sort_values(ascending=False).index
            )

            # --- Dot plot of top genes across cell types (signed logFC) ---
            if order_genes.size == 0:
                print("Ordered gene matrix is empty; skipping dotplot.")
            else:
                vmax = np.nanmax(np.abs(order_genes.values))
                vmax = 0.5 if vmax < 0.5 else vmax
                if np.isnan(vmax) or vmax == 0:
                    vmax = 0.5

                order_genes = order_genes.copy()
                order_genes.index.name = "gene"

                plot_df = order_genes.reset_index().melt(
                    id_vars="gene",
                    var_name="cell_type",
                    value_name="logFC"
                )

                plot_df["abs_logFC"] = plot_df["logFC"].abs()
                plot_df = plot_df[plot_df["abs_logFC"] > 0.0]

                gene_order = list(order_genes.index)
                ct_order   = list(order_genes.columns)

                gene_to_idx = {g: i for i, g in enumerate(gene_order)}
                ct_to_idx   = {c: i for i, c in enumerate(ct_order)}

                plot_df["x"] = plot_df["cell_type"].map(ct_to_idx)
                plot_df["y"] = plot_df["gene"].map(gene_to_idx)

                fig_h = 0.3 * len(gene_order) + 3
                fig_w = 0.5 * len(ct_order) + 6

                fig, ax = plt.subplots(figsize=(fig_w, fig_h))

                size_factor = 500 / vmax if vmax > 0 else 500

                scatter = ax.scatter(
                    x=plot_df["x"],
                    y=plot_df["y"],
                    s=plot_df["abs_logFC"] * size_factor,
                    c=plot_df["logFC"],
                    cmap="PuOr",
                    vmin=-vmax,
                    vmax=vmax,
                    edgecolors="gray",
                    alpha=0.9
                )

                cbar = fig.colorbar(scatter, ax=ax, orientation="vertical", shrink=0.7)
                cbar.set_label("log2 fold change")

                ax.set_xticks(range(len(ct_order)))
                ax.set_xticklabels(ct_order, rotation=45, ha="right",
                                   rotation_mode="anchor", fontsize=7)

                ax.set_yticks(range(len(gene_order)))
                ax.set_yticklabels(gene_order, fontsize=6)

                ax.set_xlabel("Cell Type", fontsize=10)
                ax.set_ylabel("Gene", fontsize=10)
                ax.set_title("Pseudobulk DGE: Top Genes Across Cell Types (Dot Plot)",
                             fontsize=12)

                legend_handles = [
                    ax.scatter([], [], s=vmax * size_factor,   color="gray", alpha=0.6,
                               label=f"Max ({vmax:.1f})"),
                    ax.scatter([], [], s=vmax/2 * size_factor, color="gray", alpha=0.6,
                               label=f"Med ({vmax/2:.1f})"),
                    ax.scatter([], [], s=vmax/4 * size_factor, color="gray", alpha=0.6,
                               label="Low")
                ]
                ax.legend(handles=legend_handles,
                          title=r"Effect Size (|$log_{2}FC$|)",
                          bbox_to_anchor=(1.05, 1),
                          loc="upper left",
                          fontsize=7)

                fig.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, "top_genes_dotplot.png"), dpi=300)
                plt.close(fig)
