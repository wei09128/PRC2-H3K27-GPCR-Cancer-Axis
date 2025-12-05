import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/mnt/f/H3K27/Data/TF/analysis/tf_differential_activity_merged.csv")

df_plot = df[df["group"] == "Aggressive"].copy()
df_plot["abs_lfc"] = df_plot["logfoldchanges"].abs()
df_plot = df_plot.sort_values("abs_lfc", ascending=True)

colors = np.where(df_plot["logfoldchanges"] > 0, "tab:purple", "tab:orange")

plt.figure(figsize=(6, 5))
plt.hlines(
    y=df_plot["names"],
    xmin=0,
    xmax=df_plot["logfoldchanges"],
    color=colors,
    linewidth=2
)
plt.scatter(
    df_plot["logfoldchanges"],
    df_plot["names"],
    s=40,
    color=colors,
    zorder=3
)

plt.axvline(0, color="black", linewidth=1)
plt.xlabel("log2 fold-change (Aggressive vs less-Aggressive)")
plt.ylabel("")
plt.title("TF activity differences defining aggressiveness")
plt.tight_layout()

out_png = "/mnt/f/H3K27/Data/TF/analysis/tf_activity_diverging_lollipop.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")  # <- 300 dpi PNG
print("Saved:", out_png)

plt.show()
