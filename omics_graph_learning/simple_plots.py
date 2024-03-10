#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import os
import pickle
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def _set_matplotlib_publication_parameters() -> None:
    """Set matplotlib parameters for publication quality plots."""
    font_size = 7
    plt.rcParams.update(
        {
            "font.family": "sans-serif",  # set the default font family to sans-serif
            "font.sans-serif": "Arial",  # first preference for sans-serif should be Arial
            "font.size": font_size,  # set font size
            "axes.titlesize": font_size,  # set title size for axes
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )


def place_holder_function() -> None:
    """_summary_ of function"""
    pass


def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    _set_matplotlib_publication_parameters()
    ### plot GNN architecture performances
    performances = [0.53, 0.71, 0.75]
    models = ("GCN", "GraphSAGE", "GAT")
    x_pos = np.arange(len(models))

    plt.tight_layout()
    plt.bar(x_pos, performances, align="center", alpha=0.5)
    plt.ylim(0, 1)
    plt.xticks(x_pos, models)
    plt.ylabel("Pearson correlation on test set")
    plt.title("Performance of different GNN architectures")
    plt.show()

    _set_matplotlib_publication_parameters()
    ### plot performances across feat_windows
    performances = [0.53, 0.71, 0.75]
    models = ("5kb", "10kb", "25kb", "50kb")
    x_pos = np.arange(len(models))

    plt.tight_layout()
    plt.bar(x_pos, performances, align="center", alpha=0.5)
    plt.ylim(0, 1)
    plt.xticks(x_pos, models)
    plt.ylabel("Pearson correlation on test set")
    plt.title("Performance of different GNN architectures")
    plt.show()
