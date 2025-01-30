#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code for figure 3.

    1. Produces expected vs predicted plots for each sample.
    2. Produces expected vs predicted facet plot.
    3. Produces expected vs predicted correlation heatmap, where expected and
       predicted are measured as difference from the average expression.
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
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
import torch

from omics_graph_learning.interpret.interpret_utils import get_gene_idx_mapping
from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import load_gencode_lookup
from omics_graph_learning.interpret.interpret_utils import map_symbol
from omics_graph_learning.visualization import set_matplotlib_publication_parameters
from omics_graph_learning.visualization.training import plot_predicted_versus_expected


def get_gene_idx_mapping(idxs: Dict[str, int]) -> Tuple[Dict[int, str], List[int]]:
    """Map 'ENSG...' nodes to a dict of node_idx->gene_id and a list of gene_indices."""
    gene_idxs = {k: v for k, v in idxs.items() if "ENSG" in k}
    node_idx_to_gene_id = {v: k for k, v in gene_idxs.items()}
    gene_indices = list(gene_idxs.values())
    return node_idx_to_gene_id, gene_indices


def get_top_gene_connections_fast(attn_mapped, N=100):
    # Initialize storage for each gene (target)
    sources_per_gene = defaultdict(list)
    values_per_gene = defaultdict(list)

    # Group by gene target
    for (source, target), attention_value in attn_mapped.items():
        if "ENSG" in target:  # only process if target is a gene
            sources_per_gene[target].append(source)
            values_per_gene[target].append(attention_value)

    result = {}
    for gene_target in sources_per_gene:
        values = np.array(values_per_gene[gene_target])
        sources = np.array(sources_per_gene[gene_target])

        if len(values) <= N:
            top_indices = np.argsort(values)[::-1]
        else:
            top_indices = np.argpartition(values, -N)[-N:]
            top_indices = top_indices[np.argsort(values[top_indices])][::-1]

        result[gene_target] = {sources[i]: values[i] for i in top_indices}

    return result


def main() -> None:
    """Main function."""
    attn = "/Users/steveho/gnn_plots/interpretation/k562_release/attention_weights_genes.pt"
    idxs = (
        "/Users/steveho/gnn_plots/graph_resources/idxs/k562_release_full_graph_idxs.pkl"
    )

    attn_weights = torch.load(attn)
    idxs = pickle.load(open(idxs, "rb"))

    # reverse map idxs
    idxs = {v: k for k, v in idxs.items()}

    # map idx tuples in attn_weights back to genes and stuff
    attn_mapped = {}
    for (idx_1, idx_2), val in attn_weights.items():
        node_1 = idxs[idx_1].strip("_k562")
        node_2 = idxs[idx_2].strip("_k562")
        attn_mapped[(node_1, node_2)] = val

    # get top gene connections
    top_gene_connections = get_top_gene_connections_fast(attn_mapped, N=100)


if __name__ == "__main__":
    main()
