#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Make plots related to GNN training."""


from pathlib import Path
from typing import Dict, List, Union

import matplotlib  # type: ignore
from matplotlib.colors import LogNorm  # type: ignore
from matplotlib.figure import Figure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator  # type: ignore

from omics_graph_learning.visualization import \
    set_matplotlib_publication_parameters


def _load_tb_loss(tensorboard_log: Union[str, Path]) -> Dict[str, List[float]]:
    """Load tensorboard log and return a dataframe with the loss values."""
    event_acc = EventAccumulator(str(tensorboard_log))
    event_acc.Reload()

    # get the loss values
    training_loss = event_acc.Scalars("Training loss")
    test_rmse = event_acc.Scalars("Test RMSE")

    # remove epoch 0
    val_rmse = event_acc.Scalars("Validation RMSE")
    val_rmse = [rmse for rmse in val_rmse if rmse.step > 0]

    return {
        "Epoch": [loss.step for loss in training_loss],
        "Training loss": [loss.value for loss in training_loss],
        "Validation RMSE": [rmse.value for rmse in val_rmse],
        "Test RMSE": [rmse.value for rmse in test_rmse],
    }


def plot_training_losses(tensorboard_log: Union[str, Path]) -> Figure:
    """Plots training losses. Get values from tensorboard log."""
    losses = _load_tb_loss(tensorboard_log)
    df = pd.DataFrame(losses)
    set_matplotlib_publication_parameters()

    fig, ax = plt.subplots()
    sns.lineplot(x="Epoch", y="Training loss", data=df, label="Training Loss", ax=ax)
    sns.lineplot(
        x="Epoch", y="Validation RMSE", data=df, label="Validation RMSE", ax=ax
    )
    sns.lineplot(x="Epoch", y="Test RMSE", data=df, label="Test RMSE", ax=ax)
    ax.margins(x=0)
    ax.set_xlabel("Epoch", fontsize=7)
    ax.set_ylabel("Loss / RMSE", fontsize=7)
    ax.set_title("Training and Validation Metrics", fontsize=7)
    fig.tight_layout()
    return fig


def plot_predicted_versus_expected(
    predicted: np.ndarray,
    expected: np.ndarray,
    rmse: float,
    save_path: Union[str, Path] = None,
    title: bool = True,
) -> Figure:
    """Plots predicted versus expected values for a given model"""
    set_matplotlib_publication_parameters()

    # calculate Pearson and Spearman correlations
    pearson_r, _ = stats.pearsonr(expected, predicted)
    spearman_r, _ = stats.spearmanr(expected, predicted)

    # jointplot - hexbin scatter with marginal histograms
    plot = sns.jointplot(
        x=predicted,
        y=expected,
        kind="hex",
        height=3,
        ratio=4,
        space=0.1,
        joint_kws={
            "gridsize": 30,
            "cmap": "viridis",
            "norm": LogNorm(),
            "edgecolors": "white",
            "linewidths": 0.025,
        },
        marginal_kws={
            "element": "step",
            "color": "lightsteelblue",
            "edgecolors": "lightslategray",
            "linewidth": 0.5,
        },
    )

    # set labels and title
    plot.ax_joint.set_xlabel("Predicted Log2 Expression")
    plot.ax_joint.set_ylabel("Expected Log2 Expression")
    if title:
        plot.figure.suptitle(
            f"Predicted versus expected TPM\n"
            rf"Spearman's $\rho$: {spearman_r:.4f}"
            f"\nRMSE: {rmse:.4f}",
            y=0.95,
        )

    # add colorbar
    plot.figure.colorbar(
        plot.ax_joint.collections[0],
        ax=plot.ax_joint,
        aspect=5,
        shrink=0.35,
    )

    # adjust axis limits to include all data points
    plot.ax_joint.set_xlim(np.min(predicted) - 0.5, np.max(predicted) + 0.5)
    plot.ax_joint.set_ylim(np.min(expected) - 0.5, np.max(expected) + 0.5)

    # calculate and plot the linear regression line
    slope, intercept, _, _, _ = stats.linregress(predicted, expected)
    x_fit = np.linspace(np.min(predicted) - 0.5, np.max(predicted) + 0.5, 100)
    y_fit = slope * x_fit + intercept
    plot.ax_joint.plot(
        x_fit,
        y_fit,
        color="indianred",
        linewidth=0.9,
        linestyle="--",
    )

    # add a best-fit line at 45 degrees
    min_val = min(plot.ax_joint.get_xlim()[0], plot.ax_joint.get_ylim()[0])
    max_val = max(plot.ax_joint.get_xlim()[1], plot.ax_joint.get_ylim()[1])
    plot.ax_joint.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="lightgray",
        linewidth=0.9,
        label="Perfect Fit",
    )

    # add pearson R
    plot.ax_joint.text(
        0.025,
        1.05,
        r"$\mathit{r}$ = " + f"{pearson_r:.4f}",
        transform=plot.ax_joint.transAxes,
        fontsize=7,
        verticalalignment="top",
    )
    plt.tight_layout()
    plot.savefig(save_path / "performance.png", dpi=450, bbox_inches="tight")
    return plot.figure
