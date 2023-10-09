#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import csv
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns


def _set_matplotlib_publication_parameters() -> None:
    plt.rcParams.update({"font.size": 7})  # set font size
    plt.rcParams["font.family"] = "Helvetica"  # set font


def plot_training_losses(
    log: str,
    model: str,
    layers: int,
    width: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    losses = {"Train": [], "Test": [], "Validation": []}
    with open(log, newline="") as file:
        reader = csv.reader(file, delimiter=":")
        for line in reader:
            for substr in line:
                for key in losses:
                    if key in substr:
                        losses[key].append(float(line[-1].split(" ")[-1]))

    losses = pd.DataFrame(losses)
    plt.figure(figsize=(3, 2.25))
    sns.lineplot(data=losses)
    plt.margins(x=0)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(
        f"Training loss for {model}, {layers} layers, lr {learning_rate}, batch size {batch_size}, dimensions {width}",
        wrap=True,
    )
    plt.tight_layout()
    plt.savefig(
        f"{model}_{layers}_{width}_{batch_size}_{learning_rate}_loss.png",
        dpi=300,
    )
    plt.close()


def plot_predicted_versus_expected(expected, predicted, model, layers, width):
    sns.regplot(x=expected, y=predicted, scatter_kws={"s": 2, "alpha": 0.1})
    plt.margins(x=0)
    plt.xlabel("Expected Log2 TPM")
    plt.ylabel("Predcited Log2 TPM")
    # plt.title(f"{model}")
    plt.title(f"Remove H3K27ac features")

    res = stats.spearmanr(expected, predicted)


def main(plot_dir: str) -> None:
    """Main function"""
    _set_matplotlib_publication_parameters()

    log = "regulatoryonly_combinedloops_GraphSAGE_2_256_0.001_batch256_neighbor_full_targetnoscale_idx_expression_only.log"
    model = "GraphSAGE"
    layers = 2
    width = 256
    batch_size = 256
    learning_rate = 0.001


if __name__ == "__main__":
    main(
        plot_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/models/plots"
    )
