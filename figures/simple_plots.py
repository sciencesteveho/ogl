#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import os
import pickle
import random
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _set_matplotlib_publication_parameters() -> None:
    """Set matplotlib parameters for publication quality plots."""
    font_size = 7
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": "Arial",
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )


def plot_performance(
    performances: List[float],
    models: Tuple[str, ...],
    title: str,
    ylabel: str,
    highlight_first: bool = False,
) -> None:
    """
    Plot the performance of different models.

    Args:
        performances: A list of performance values.
        models: A tuple of model names.
        title: The title of the plot.
        ylabel: The label for the y-axis.

    Returns:
        None
    """
    x_pos = np.arange(len(models))
    colors = ["blue"] * len(models)

    if highlight_first:
        colors[0] = "red"

    plt.tight_layout()
    plt.bar(x_pos, performances, align="center", alpha=0.5, color=colors)
    for i, v in enumerate(performances):
        plt.text(i, v + 0.01, str(round(v, 2)), ha="center")
    plt.ylim(0, 1)
    plt.xticks(x_pos, models)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_histogram_from_file(filename: str, title: str) -> None:
    """
    Plot a histogram of 100k randomly sampled from distances

    Args:
        filename: The name of the file containing the distances.
        title: The title of the histogram plot.

    Returns:
        None
    """
    with open(filename, "r") as file:
        distances = [int(line) for line in file.readlines()]
    distances = random.sample(distances, 100000)
    plt.hist(distances, bins=74)
    plt.xlim(0, 2000000)
    plt.title(title)
    plt.xlabel("Genomic distance (bp)")
    plt.ylabel("Frequency")
    plt.show()


def main() -> None:
    """Plot figs!"""
    _set_matplotlib_publication_parameters()

    # deeploop contacts
    plot_performance(
        [0.75, 0.43, 0.42, 0.42],
        ("300K", "1 million", "2 million", "3 million"),
        "Performance increasing number of contacts from DeepLoop",
        "Pearson correlation on test set",
        highlight_first=True,
    )

    # gnn architectures
    plot_performance(
        [0.53, 0.71, 0.75],
        ("GCN", "GraphSAGE", "GAT"),
        "Performance of different GNN architectures",
        "Pearson correlation on test set",
    )

    # feat windows
    plot_performance(
        [0.62, 0.65, 0.60, 0.68, 0.63, 0.69],
        ("2kb", "5kb", "10kb", "25kb", "50kb", "100kb"),
        "Performance when changing feature windows",
        "Spearmann correlation on test set",
    )

    # HiCDC+ FDR for k562 and left ventricle
    plot_performance(
        [0.40, 0.46, 0.51],
        ("FDR 0.1", "FDR 0.01", "FDR 0.001"),
        "Performance across HiCDC+ FDR Cutoffs for K562",
        "Pearson correlation on test set",
    )
    plot_performance(
        [0.36, 0.39, 0.41],
        ("FDR 0.1", "FDR 0.01", "FDR 0.001"),
        "Performance across HiCDC+ FDR Cutoffs for Left Ventricle",
        "Spearmann correlation on test set",
    )

    # regulatory schema
    plot_performance(
        [0.52, 0.55, 0.57, 0.61],
        ("encode", "epimap", "union", "intersect"),
        "Performance across different regulatory schemes",
        "Pearson correlation on test set",
    )

    # histograms for distance between anchors for deeploop contacts and for hi-c
    # contacts
    plot_histogram_from_file(
        "deeploop_distances.txt",
        "Distribution of distances between 100K randomly sampled DeepLoop contacts",
    )
    plot_histogram_from_file(
        "hic_distances.txt",
        "Distribution of distances between 100K randomly sampled Hi-C contacts at FDR 0.001",
    )


if __name__ == "__main__":
    main()


"""
awk '{
    if ($1 == $4) {
        start = $3 < $6 ? $3 : $6;
        end = $3 > $6 ? $3 : $6;
        distance = end - start;
        print (distance >= 0 ? distance : -distance);
    } else {
        print "Different chromosomes";
    }
}' GSE167200_LeftVentricle.top300K.txt > deeploop_distances.txt

awk '{
    if ($1 == $4) {
        start = $3 < $6 ? $3 : $6;
        end = $3 > $6 ? $3 : $6;
        distance = end - start;
        print (distance >= 0 ? distance : -distance);
    } else {
        print "Different chromosomes";
    }
}' leftventricle_all_chr_0.001.tsv > hic_distances.txt
"""
