#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code for figure node feature ablations.
"""

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
    32: "Rep G2",
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


def plot_average_effect(df, tissue):
    """
    Plots the average 'effect' for each node index in the dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame containing at least 'node' and 'effect' columns.
    """
    avg_effect = df.groupby("node")["effect"].mean().reset_index()

    # map node indices to feature names (fallback to the node index as string)
    avg_effect["feature"] = (
        avg_effect["node"].map(FEATURES).fillna(avg_effect["node"].astype(str))
    )

    # sort the dataframe by the effect values
    avg_effect.sort_values("effect", inplace=True)

    # calculate symmetrical limits around zero
    min_val = avg_effect["effect"].min()
    max_val = avg_effect["effect"].max()
    abs_max = max(abs(min_val), abs(max_val))

    # set up normalization for the colormap based on the effect range
    norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)

    # use the 'coolwarm' colormap where negative values appear blue and positive values red
    colors = plt.cm.coolwarm(norm(avg_effect["effect"]))

    plt.figure(figsize=(3.5, 5))
    bars = plt.barh(avg_effect["feature"], avg_effect["effect"], color=colors)
    plt.xlabel("Average effect")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.5)  # plot vertical line

    plt.tight_layout()
    plt.savefig(f"effect_size/{tissue}_average_effect_per_node.png")
    plt.clf()
    plt.close()


def plot_average_effect_for_gene(df, tissue, gene_id):
    """
    Plots the average 'effect' for a specific gene (gene_id) within a given tissue.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'gene', 'node', and 'effect' columns.
    tissue : str
        Name of the tissue (used for the output filename).
    gene_id : str
        The gene identifier (e.g., "ENSG00000240972.1") to plot.
    """
    # Filter the DataFrame to the specified gene
    gene_df = df[df["gene"] == gene_id].copy()
    if gene_df.empty:
        print(f"No data found for gene {gene_id} in {tissue}.")
        return

    # Group by 'node' and compute mean effect
    avg_effect = gene_df.groupby("node")["effect"].mean().reset_index()

    # Map node indices to feature names (fallback to the node index as string)
    avg_effect["feature"] = (
        avg_effect["node"].map(FEATURES).fillna(avg_effect["node"].astype(str))
    )

    # Sort the dataframe by the effect values
    avg_effect.sort_values("effect", inplace=True)

    # Calculate symmetrical limits around zero
    min_val = avg_effect["effect"].min()
    max_val = avg_effect["effect"].max()
    abs_max = max(abs(min_val), abs(max_val))

    # Set up normalization for the colormap based on the effect range
    norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)

    # Use the 'coolwarm' colormap where negative values appear blue and positive values red
    colors = plt.cm.coolwarm(norm(avg_effect["effect"]))

    # Create a horizontal bar plot
    plt.figure(figsize=(3.5, 5))
    plt.barh(avg_effect["feature"], avg_effect["effect"], color=colors)

    # Label axes and draw a vertical line at x=0
    plt.xlabel("Average effect")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"effect_size/{tissue}_{gene_id}_average_effect_per_node.png")
    plt.clf()
    plt.close()


def find_negative_index_keys(data, index):
    result = []
    for outer_key, outer_value in data.items():
        for inner_key, inner_value in outer_value.items():
            if index in inner_value and inner_value[index] < 0:
                result.append(
                    {
                        "outer_key": outer_key,
                        "inner_key": inner_key,
                        "value": inner_value[index],
                    }
                )
    return result


set_matplotlib_publication_parameters()
all_tissue_effects = []  # will store dicts with {tissue, node, feature, avg_effect}


for tissue in TISSUES:
    perturb_file = f"{tissue}_release/selected_component_perturbations.pkl"
    with open(perturb_file, "rb") as f:
        perts = pickle.load(f)

    # convert to dataframe
    data = []
    for gene, reg_dict in perts.items():
        for reg_elem, node_dict in reg_dict.items():
            data.extend(
                {
                    "gene": gene,
                    "regulatory_element": reg_elem,
                    "node": node,
                    "effect": effect,
                }
                for node, effect in node_dict.items()
            )

    df = pd.DataFrame(data)
    df["gene"] = df["gene"].str.replace(f"_{tissue}", "")
    plot_average_effect(df, tissue)

    # Compute average effect per node for this tissue
    avg_effect = df.groupby("node")["effect"].mean().reset_index()
    avg_effect["feature"] = (
        avg_effect["node"].map(FEATURES).fillna(avg_effect["node"].astype(str))
    )

    # Store in all_tissue_effects
    for row in avg_effect.itertuples():
        all_tissue_effects.append(
            {
                "tissue": tissue,
                "node": row.node,
                "feature": row.feature,
                "avg_effect": row.effect,
            }
        )

df_all = pd.DataFrame(all_tissue_effects)
heatmap_df = df_all.pivot(index="tissue", columns="feature", values="avg_effect")

# Optional: rename the row index using TISSUES mapping for a nicer label
heatmap_df.rename(index=TISSUES, inplace=True)

colors = [
    "#08306b",
    "#083c9c",
    "#08519c",
    "#3182bd",
    "#7cabf7",
    "#81cffc",
    "#ffffff",
    "#fc9179",
    "#fb6a4a",
    "#fb504a",
    "#db2727",
    "#9c0505",
    "#99000d",
]

custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=100)

sns.heatmap(
    heatmap_df,
    cmap=custom_cmap,
    center=0,  # ensures 0 is the midpoint of the color scale
    cbar_kws={"label": "Average Effect", "shrink": 0.2, "aspect": 10},
    square=True,
)
plt.title("Average effect per feature across 100 most impactful elements")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig("effect_size/average_effects_heatmap.png", bbox_inches="tight")
plt.clf()
plt.close()

# cell lines only
KEEP = ["K562", "GM12878", "H1-hESC", "HepG2", "IMR90", "NHEK", "HMEC"]
KEEP = sorted(KEEP)

cell_df = heatmap_df.loc[KEEP]
cell_df = cell_df.transpose()

sns.heatmap(
    cell_df,
    cmap=custom_cmap,
    center=0,  # ensures 0 is the midpoint of the color scale
    cbar_kws={"label": "Average Effect", "shrink": 0.2, "aspect": 8},
    square=True,
)

plt.title("Average effect per feature across 100 most impactful elements")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig("effect_size/average_effects_heatmap_cell_lines.png", bbox_inches="tight")
plt.clf()
plt.close()


# atac_keys = find_negative_index_keys(test, 7)
# print(atac_keys)

# k27ac_keys = find_negative_index_keys(test, 12)
# print(k27ac_keys)


# df = pd.DataFrame(data)
# # remove _k562 from gene names
# df["gene"] = df["gene"].str.replace("_k562", "")

gene_df = df[df["gene"] == "ENSG00000141510.16"]

df["abs_effect"] = df["effect"].abs()
top = df.nlargest(100, "abs_effect")


# plot_average_effect(df)
