#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code for figure 3.
"""


from collections import defaultdict
import csv
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union

import matplotlib  # type: ignore
from matplotlib.collections import PolyCollection  # type: ignore
from matplotlib.colors import LogNorm  # type: ignore
import matplotlib.colors as mcolors
from matplotlib.figure import Figure  # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from numpy import sqrt
import numpy as np
import pandas as pd
import pybedtools
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
import torch
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.interpret_utils import get_gene_idx_mapping
from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import load_gencode_lookup
from omics_graph_learning.interpret.interpret_utils import map_symbol
from omics_graph_learning.visualization import set_matplotlib_publication_parameters
from omics_graph_learning.visualization.training import plot_predicted_versus_expected

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


def sort_by_type(idxs: Dict[str, int]) -> Tuple[Dict[int, int], List[int]]:
    """Sort an idx file into different types of nodes and return the new order
    for better visualization.
    """
    # split into different types
    ensg = {k: v for k, v in idxs.items() if k.startswith("ENSG")}
    promoter = {k: v for k, v in idxs.items() if "promoter" in k}
    dyadic = {k: v for k, v in idxs.items() if "dyadic" in k}
    enhancer = {k: v for k, v in idxs.items() if "enhancer" in k}

    # create a mapping from old indices to new positions
    old_to_new = {}
    current_pos = 0

    # add them in specific order
    for group in [ensg, promoter, dyadic, enhancer]:
        for k in sorted(group.keys()):
            old_to_new[group[k]] = current_pos
            current_pos += 1

    # calculate separation points
    separation_points: List[int] = [
        len(ensg),
        len(ensg) + len(promoter),
        len(ensg) + len(promoter) + len(dyadic),
    ]

    return old_to_new, separation_points


def plot_sample_level_saliency(
    tissue: str,
    saliency_map: str,
    idx_file: str,
    outpath: str,
) -> None:
    """Plot the iunput x gradient saliency attribution map global view for a
    sample.
    """
    # load idxs
    with open(idx_file, "rb") as f:
        idxs = pickle.load(f)

    # sort by type
    old_to_new, separation_points = sort_by_type(idxs)

    # load and reorder saliency map
    saliency = torch.load(saliency_map)
    saliency_np = (
        saliency.detach().cpu().numpy() if hasattr(saliency, "detach") else saliency
    )
    saliency_reordered = np.zeros_like(saliency_np)
    for old_idx, new_idx in old_to_new.items():
        saliency_reordered[new_idx] = saliency_np[old_idx]

    # plot saliency heatmap
    fix, ax = plt.subplots()
    sns.heatmap(saliency_reordered, cmap="viridis", cbar=False, ax=ax)  # type: ignore

    cbar = plt.colorbar(
        ax.collections[0],
        shrink=0.25,
        aspect=7.5,
        ax=ax,
        location="left",
        pad=0.08,
    )
    cbar.outline.set_linewidth(0.5)  # type: ignore
    cbar.ax.set_title("Contribution", pad=10)

    # annotate node classes
    n_rows = saliency_reordered.shape[0]
    n_cols = saliency_reordered.shape[1]
    section_starts = [0] + separation_points
    section_ends = separation_points + [n_rows]
    labels = ["genes", "promoters", "dyadic", "enhancers"]

    ax.set_xlim(-0.5, n_cols + 4)  # Add small margin for brackets and text
    bracket_gap = 0.25

    for i in range(len(section_starts)):
        start = section_starts[i]
        end = section_ends[i]
        midpoint = (start + end) / 2

        # bracket coordinates
        x_start = n_cols + 0.5
        x_bracket = n_cols + 1.5

        ax.plot([x_start, x_bracket], [start, start], "k-", linewidth=0.5)  # Top line
        ax.plot([x_start, x_bracket], [end, end], "k-", linewidth=0.5)  # Bottom line

        # vertical line
        ax.plot(
            [x_bracket, x_bracket],
            [start, midpoint + bracket_gap / 2],
            "k-",
            linewidth=0.5,
        )
        ax.plot(
            [x_bracket, x_bracket],
            [midpoint + bracket_gap / 2, end],
            "k-",
            linewidth=0.5,
        )

        # add node class labels
        ax.text(x_bracket + 0.5, midpoint, labels[i], va="center", ha="left")

    # add feature labels
    xtick_positions = np.arange(5, len(saliency_np[0])) + 0.5
    xtick_labels = [FEATURES[i] for i in range(5, 42)]
    plt.xticks(xtick_positions, xtick_labels, rotation=90, ha="center")
    plt.yticks([])

    # add positional encoding bracket for first 5 features
    ax.text(
        2.3,
        -0.03,
        "positional\nencoding",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
    )

    # add plot descriptors
    plt.xlabel("Features")
    plt.ylabel("Nodes")
    plt.title(f"{tissue} input x gradient saliency map")

    plt.tight_layout()
    plt.savefig(f"{outpath}/saliency_map.png", dpi=450, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_selected_saliency(
    tissue: str,
    saliency_map: str,
    idx_file: str,
    outpath: str,
    selected_idxs: Union[List[int], range],
    node_type: str = "genes",
) -> None:
    """Plot the input x gradient saliency attribution map for a specified set of
    nodes.

    The saliency map is first reordered (using sort_by_type) and then a subset
    corresponding to the provided original (pre-reordering) indices is selected.
    The y-axis is labeled with the original indices.
    """
    import pickle
    from typing import List, Union

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch

    # load indices
    with open(idx_file, "rb") as f:
        idxs = pickle.load(f)

    # sort indices by type
    old_to_new, separation_points = sort_by_type(idxs)

    # load and reorder saliency map
    saliency = torch.load(saliency_map)
    saliency_np = (
        saliency.detach().cpu().numpy() if hasattr(saliency, "detach") else saliency
    )
    saliency_reordered = np.zeros_like(saliency_np)
    for old_idx, new_idx in old_to_new.items():
        saliency_reordered[new_idx] = saliency_np[old_idx]

    # map the provided original indices to their new positions.
    selected_mapping = [(old_to_new[i], i) for i in selected_idxs if i in old_to_new]
    if not selected_mapping:
        raise ValueError(
            "None of the selected indices were found in the index mapping."
        )

    # sort the selected rows by their new (reordered) index so the plot reflects
    # the ordering.
    selected_mapping.sort(key=lambda x: x[0])
    new_indices, original_labels = zip(*selected_mapping)

    # extract saliency only for the selected indices
    saliency_selected = saliency_reordered[np.array(new_indices), :]

    # plot saliency heatmap for selected indices
    fig, ax = plt.subplots()
    sns.heatmap(saliency_selected, cmap="viridis", cbar=False, ax=ax)  # type: ignore

    # add colorbar
    cbar = plt.colorbar(
        ax.collections[0],
        shrink=0.25,
        aspect=7.5,
        ax=ax,
        location="left",
        pad=0.08,
    )
    cbar.outline.set_linewidth(0.5)  # type: ignore
    cbar.ax.set_title("Contribution", pad=10)

    # add feature labels
    n_cols = saliency_np.shape[1]
    xtick_positions = np.arange(5, n_cols) + 0.5
    xtick_labels = [FEATURES[i] for i in range(5, n_cols)]
    plt.xticks(xtick_positions, xtick_labels, rotation=90, ha="center")
    plt.yticks([])

    # add positional encoding bracket for first 5 features
    ax.text(
        2.3,
        -0.03,
        "positional\nencoding",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
    )

    # plot descriptors
    plt.xlabel("Features")
    plt.ylabel("Nodes")
    plt.title(f"{tissue} input x gradient saliency map ({node_type})")

    plt.tight_layout()
    plt.savefig(f"{outpath}/saliency_map_{node_type}.png", dpi=450, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_avg_saliency_heatmap(df: pd.DataFrame) -> None:
    """Plot the average saliency scores per node feature as an annotated heatmap
    with violin plots that illustrate the distribution of scores across tissues.
    """
    set_matplotlib_publication_parameters()
    df = df.iloc[:, 5:]
    df = df.rename(columns=FEATURES)

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)
    ax_violin = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # get colors
    n_features = len(df.columns)
    feature_colors = plt.cm.tab20(np.linspace(0, 1, n_features))  # type: ignore

    # melt df
    df_melt = df.reset_index().melt(
        id_vars="index", var_name="Feature", value_name="AvgSaliency"
    )
    feature_order = list(df.columns)

    sns.violinplot(
        data=df_melt,
        x="Feature",
        y="AvgSaliency",
        order=feature_order,
        ax=ax_violin,
        cut=0,
        inner="box",
        density_norm="width",
        linewidth=0.5,
        palette=list(feature_colors),
        width=0.7,
        saturation=1.0,
        inner_kws=dict(box_width=2),
        hue="Feature",
        legend=False,
    )

    # clean up the violin plot axes
    ax_violin.set_xlabel("")
    ax_violin.set_ylabel("")
    ax_violin.set_xticklabels([])
    ax_violin.grid(False)
    for spine in ["top", "right", "bottom", "left"]:
        ax_violin.spines[spine].set_linewidth(0.5)
        ax_violin.spines[spine].set_color("black")

    # custom colormap cause pretty pretty
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
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=100)

    # set vmin and vmax for heatmap
    vmin = df.min().min()
    vmax = df.max().max()

    # plot heatmap
    sns.heatmap(
        df,
        cmap="viridis",
        ax=ax_heat,
        cbar_kws={
            "label": "Average saliency score",
            "shrink": 0.3,
            "aspect": 6.5,
            "orientation": "vertical",
            "pad": 0.012,
        },
        xticklabels=feature_order,
        linewidths=0.1,
        linecolor="white",
        square=True,
        vmin=vmin,
        vmax=vmax,
    )

    ax_heat.set_xticklabels(feature_order, rotation=90, ha="center", va="top", y=-0.005)

    # add colored squares to the x-axis labels corresponding to each feature
    for label, color in zip(ax_heat.get_xticklabels(), feature_colors):
        text = label.get_text()
        pos = label.get_position()
        label._text = f"{text}"
        ax_heat.text(
            pos[0],
            pos[1] + 0.04,
            "■",
            fontsize=10,
            color=color,
            transform=label.get_transform(),
            rotation=90,
            va=label.get_va(),
            ha=label.get_ha(),
        )

    # align the violin plot with the heatmap.
    heatmap_pos = ax_heat.get_position()
    violin_pos = ax_violin.get_position()
    ax_violin.set_position(
        [heatmap_pos.x0, violin_pos.y0, heatmap_pos.width, violin_pos.height]
    )

    plt.tight_layout()
    plt.savefig("average_saliency_heatmap.png", dpi=450, bbox_inches="tight")
    plt.clf()
    plt.close()


def main() -> None:
    """Main function."""
    set_matplotlib_publication_parameters()
    interp_path = "/Users/steveho/gnn_plots/interpretation"
    idx_path = "/Users/steveho/gnn_plots/graph_resources/idxs"

    # plot global saliency per tissue
    for tissue in TISSUES.keys():
        outpath = f"{interp_path}/{tissue}_release"
        saliency_map = f"{outpath}/scaled_saliency_map.pt"
        idx_file = f"{idx_path}/{tissue}_release_full_graph_idxs.pkl"
        plot_sample_level_saliency(TISSUES[tissue], saliency_map, idx_file, outpath)

    # plot selected saliency
    tissue = "k562"
    outpath = f"{interp_path}/{tissue}_release"
    saliency_map = f"{outpath}/scaled_saliency_map.pt"
    idx_file = f"{idx_path}/{tissue}_release_full_graph_idxs.pkl"
    # k562 separation points
    # [46346, 72678, 158781]
    # for k562, run selected saliency for genes, promoters, dyadic, enhancers
    for node_type in ["genes", "promoters", "dyadic", "enhancers"]:
        plot_selected_saliency(
            tissue=TISSUES["k562"],
            saliency_map=saliency_map,
            idx_file=idx_file,
            outpath=outpath,
            selected_idxs=(
                range(46346)
                if node_type == "genes"
                else (
                    range(46346, 72678)
                    if node_type == "promoters"
                    else (
                        range(72678, 158781)
                        if node_type == "dyadic"
                        else range(158781, 833700)
                    )
                )
            ),
            node_type=node_type,
        )

    # get average saliency per tissue per idx
    # combine into a single df for heatmap
    all_tissue_avgs = []
    for tissue in TISSUES:
        outpath = f"{interp_path}/{tissue}_release"
        saliency_map = f"{outpath}/scaled_saliency_map.pt"

        # load saliency map
        saliency = torch.load(saliency_map)
        saliency_np = (
            saliency.detach().cpu().numpy() if hasattr(saliency, "detach") else saliency
        )

        avg_saliency = np.mean(saliency_np, axis=0)
        all_tissue_avgs.append(avg_saliency)

    df = pd.DataFrame(all_tissue_avgs, index=list(TISSUES.values()))
    plot_avg_saliency_heatmap(df)


if __name__ == "__main__":
    main()
