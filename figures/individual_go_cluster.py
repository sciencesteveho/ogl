#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code for figure node feature ablations.
"""


import itertools
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union

from gseapy import dotplot  # type: ignore
import gseapy as gp  # type: ignore
from matplotlib.colors import LinearSegmentedColormap  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import load_gencode_lookup
from omics_graph_learning.visualization import set_matplotlib_publication_parameters

FEATURES = {
    5: "Size",
    6: "GC-content",
    7: "ATAC",
    8: "CNV",
    9: "CpG methylation",
    10: "CTCF",
    11: "DNase",
    12: "H3K27ac",
    13: "H3K27me3",
    14: "H3K36me3",
    15: "H3K4me1",
    16: "H3K4me2",
    17: "H3K4me3",
    18: "H3K79me2",
    19: "H3K9ac",
    20: "H3K9me3",
    21: "Indels",
    22: "LINE",
    23: "Long terminal repeats",
    24: "Microsatellites",
    25: "PhastCons",
    26: "POLR2A",
    27: "PolyA sites",
    28: "RAD21",
    29: "RBP binding sites",
    30: "Recombination rate",
    31: "Rep G1b",
    # 32: "Rep G2",
    33: "Rep S1",
    34: "Rep S2",
    35: "Rep S3",
    36: "Rep S4",
    37: "RNA repeat",
    38: "Simple repeats",
    39: "SINE",
    40: "SMC3",
    41: "SNP",
}

TISSUES = {
    "adrenal": "Adrenal",
    "aorta": "Aorta",
    "gm12878": "GM12878",
    "h1_esc": "H1-hESC",
    "hepg2": "HepG2",
    "hippocampus": "Hippocampus",
    "hmec": "HMEC",
    "imr90": "IMR90",
    "k562": "K562",
    "left_ventricle": "Left ventricle",
    "liver": "Liver",
    "lung": "Lung",
    "mammary": "Mammary",
    "nhek": "NHEK",
    "ovary": "Ovary",
    "pancreas": "Pancreas",
    "skeletal_muscle": "Skeletal muscle",
    "skin": "Skin",
    "small_intestine": "Small intestine",
    "spleen": "Spleen",
}

alpha = 0.7
size_range = (5, 100)
gene_set = "reactome"
feat = "H3K4me1"

gene_sets = ["reactome", "go_mf", "go_bp"]
feats = [FEATURES[feature] for feature in FEATURES]

n_bins = 150
for gene_set, feat in itertools.product(gene_sets, feats):
    filename = f"{gene_set}_{feat}.png"
    df = pd.read_csv(f"{gene_set}_{feat}.csv")

    set_matplotlib_publication_parameters()

    df["sample_display"] = df["Sample"].map(TISSUES)

    # sort terms by neg_log_p
    df = df.sort_values("neg_log_p", ascending=False).copy()

    # count number of genes
    df["gene_count"] = df["Genes"].str.split(";").str.len()

    n_samples = len(df["sample_display"].unique())
    n_terms = len(df["Term"].unique())

    # Dynamically set figure size
    # base_width = n_samples * 0.3
    # base_height = n_terms * 0.3
    # width = max(8.15, min(10.05, base_width))
    # height = max(1.8, min(3.55, base_height))

    # width=8.15
    # height=3.55

    # # Initialize plot
    # plt.figure(figsize=(width, height))

    colors = ["#13436b", "#a3d3ff"]  # Light to dark
    cmap = LinearSegmentedColormap.from_list("custom", colors[::-1], N=n_bins)

    # Scatter plot
    ax = sns.scatterplot(
        data=df,
        x="sample_display",
        y="Term",
        hue="neg_log_p",
        size="gene_count",
        palette=cmap,
        sizes=size_range,
        alpha=alpha,
        zorder=2,
    )

    ax.invert_yaxis()  # Reverse y-axis for better readability
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.set_axisbelow(True)

    # Reduce spine width
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Set ticks and limits
    plt.xticks(range(n_samples), df["sample_display"].unique(), rotation=90)
    plt.yticks(range(n_terms), df["Term"].unique())

    ax.tick_params(axis="both", length=0)  # Remove tick marks but keep labels
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(n_terms - 0.5, -0.5)

    # Create colorbar for -log10 P-value
    norm = plt.Normalize(df["neg_log_p"].min(), df["neg_log_p"].max())

    sm = plt.cm.ScalarMappable(
        norm=norm,
        cmap=LinearSegmentedColormap.from_list("custom", colors[::-1], N=n_bins),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, shrink=0.275, aspect=4)
    cbar.set_label("-log$_{10}$ adjusted $\it{P}$")
    cbar.outline.set_linewidth(0.5)  # Adjust border thickness

    # Manually create size legend
    sizes = np.linspace(
        df["gene_count"].min(), df["gene_count"].max(), num=5, dtype=int
    )  # Pick 5 size values
    size_legend = [
        plt.scatter(
            [],
            [],
            s=(
                (s - df["gene_count"].min())
                / (df["gene_count"].max() - df["gene_count"].min())
                * (size_range[1] - size_range[0])
                + size_range[0]
            ),
            color="gray",
            alpha=0.6,
        )
        for s in sizes
    ]

    plt.legend(
        size_legend,
        sizes,
        title="Gene count",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        labelspacing=0.2,
        handletextpad=0.2,
        markerscale=0.7,
    )

    # Automatically adjust aspect ratio
    # aspect_ratio = (width - 0.5) / height * (n_terms / n_samples)
    # ax.set_aspect(aspect_ratio)

    plt.tight_layout()
    plt.savefig(filename, dpi=450)
    plt.close()
    plt.clf()
