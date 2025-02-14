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


def main() -> None:
    """Main function."""


if __name__ == "__main__":
    main()

saliency = torch.load("scaled_saliency_map.pt")
set_matplotlib_publication_parameters()

# convert to array
saliency_np = (
    saliency.detach().cpu().numpy() if hasattr(saliency, "detach") else saliency
)

# create the plot
ax = sns.heatmap(saliency_np, cmap="viridis", cbar=False)
plt.colorbar(ax.collections[0], shrink=0.25, aspect=5)


# Save with high DPI
plt.tight_layout()
plt.savefig("saliency_map.png", dpi=450, bbox_inches="tight")

# Clean up
plt.clf()
plt.close()
