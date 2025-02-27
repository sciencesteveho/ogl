import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact  # type: ignore
import statsmodels.stats.multitest as smm  # type: ignore

from omics_graph_learning.visualization import set_matplotlib_publication_parameters

set_matplotlib_publication_parameters()

# Color-coding function for volcano plot
p_threshold = 0.05
log2OR_threshold = 1.0


def color_code(row):
    if row["significant"]:
        if row["log2_or"] > log2OR_threshold:
            return "red"  # Enriched in top set
        elif row["log2_or"] < -log2OR_threshold:
            return "blue"  # Enriched in bottom set
        else:
            return "orange"  # Significant but moderate effect
    else:
        return "grey"


df["color"] = df.apply(color_code, axis=1)

# -------------------------
#
# 2) Volcano Plot (Figure 1)
# -------------------------
fig1, ax_volcano = plt.subplots(figsize=(2.5, 2.5))

ax_volcano.scatter(
    df["log2_or"],
    df["neg_log10_fdr"],
    c=df["color"],
    s=10,
)

# Reference lines
ax_volcano.axvline(0, color="gray", linestyle="--", lw=0.5)
ax_volcano.axhline(-np.log10(p_threshold), color="gray", linestyle="--", lw=0.5)

ax_volcano.set_xlabel(r"$\log_{2}(\text{Odds Ratio})$")
ax_volcano.set_ylabel(r"$-\log_{10}(\text{FDR})$")
ax_volcano.set_title("Genes only", pad=15)


# remove top and right spines
ax_volcano.spines["top"].set_visible(False)
ax_volcano.spines["right"].set_visible(False)

# Label significant points above threshold
for motif_name, row in df.iterrows():
    if row["significant"] and abs(row["log2_or"]) >= log2OR_threshold:
        ax_volcano.text(
            row["log2_or"],
            row["neg_log10_fdr"] + 0.05,
            motif_name,
            ha="center",
        )

plt.tight_layout()
plt.savefig("volcano_plot_genesonly.png", dpi=450)
plt.clf()
plt.close()

# # -------------------------
# # 3) Grouped Bar Chart (Figure 2)
# # -------------------------
# fig2, ax_bar = plt.subplots(figsize=(10, 6))

# motifs = df.index
# x = np.arange(len(motifs))
# width = 0.4

# ax_bar.bar(x - width / 2, df["top_count"], width, label="Top Genes")
# ax_bar.bar(x + width / 2, df["bottom_count"], width, label="Bottom Genes")

# ax_bar.set_title("Microsatellite Motif Counts")
# ax_bar.set_xlabel("Motif")
# ax_bar.set_ylabel("Count")

# ax_bar.set_xticks(x)
# ax_bar.set_xticklabels(motifs, rotation=90, fontsize=8)

# ax_bar.legend()

# plt.tight_layout()
# plt.savefig("bar_enriched.png", dpi=450)
# plt.clf()
# plt.close()
