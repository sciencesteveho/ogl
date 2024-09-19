#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Make plots related to GNN training."""


from pathlib import Path
from typing import Dict, List, Union

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
from scipy.stats import spearmanr  # type: ignore
import seaborn as sns  # type: ignore
import torch

from omics_graph_learning.visualization import set_matplotlib_publication_parameters


def plot_training_losses(
    log: str,
) -> matplotlib.figure.Figure:
    """Plots training losses from training log"""
    plt.figure(figsize=(3.125, 2.25))
    set_matplotlib_publication_parameters()

    losses: Dict[str, List[float]] = {"Train": [], "Test": [], "Validation": []}
    with open(log, newline="") as file:
        reader = csv.reader(file, delimiter=":")
        for line in reader:
            for substr in line:
                for key in losses:
                    if key in substr:
                        losses[key].append(float(line[-1].split(" ")[-1]))

    # # remove last item in train
    # try:
    #     loss_df = pd.DataFrame(losses)
    # except ValueError:
    #     losses["Train"] = losses["Train"][:-1]
    if len(losses["Train"]) > len(losses["Test"]):
        losses["Train"] = losses["Train"][:-1]

    sns.lineplot(data=losses)
    plt.margins(x=0)
    plt.xlabel("Epoch", fontsize=7)
    plt.ylabel("MSE Loss", fontsize=7)
    plt.title(
        "Training loss",
        wrap=True,
        fontsize=7,
    )
    plt.tight_layout()
    return plt


def plot_predicted_versus_expected(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    rmse: torch.Tensor,
) -> matplotlib.figure.Figure:
    """Plots predicted versus expected values for a given model"""
    plt.figure(figsize=(3.15, 2.95))
    set_matplotlib_publication_parameters()

    sns.regplot(x=expected, y=predicted, scatter_kws={"s": 2, "alpha": 0.1})
    plt.margins(x=0)
    plt.xlabel("Expected Log2 TPM", fontsize=7)
    plt.ylabel("Predicted Log2 TPM", fontsize=7)
    plt.title(
        f"Expected versus predicted TPM\n"
        f"RMSE: {rmse}\n"
        f"Spearman's R: {stats.spearmanr(expected, predicted)[0]}\n"
        f"Pearson: {stats.pearsonr(expected, predicted)[0]}",
        wrap=True,
        fontsize=7,
    )
    plt.tight_layout()
    return plt
