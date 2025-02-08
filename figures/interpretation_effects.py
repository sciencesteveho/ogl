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

TISSUES = [
    "adrenal",
    "aorta",
    "gm12878",
    "h1_esc",
    "hepg2",
    "hippocampus",
    "hmec",
    "imr90",
    "k562",
    "left_ventricle",
    "liver",
    "lung",
    "mammary",
    "nhek",
    "ovary",
    "pancreas",
    "skeletal_muscle",
    "skin",
    "small_intestine",
    "spleen",
]


def get_top_gene_connections_fast(attn_mapped, N=100):
    """Given a dictionary of attention weights, return the top N connections for
    each gene.
    """
    # initialize storage for each gene (target)
    sources_per_gene = defaultdict(list)
    values_per_gene = defaultdict(list)

    # group by gene target
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


def get_max_perturbation_effect(
    tissue: str, perturbations: Dict[str, Dict[str, float]]
):
    """Given a tissue, go through each gene, storing the perturbation effect for
    an enhancer if it does not yet exist, or updating it if the new perturbation
    effect is larger.

    The pertubation effects are in the form of
        {gene: source{'fold_change': float}}}
    """
    max_effects = {}
    working_dir = "/Users/steveho/gnn_plots/interpretation"

    # First loop: Build the nested structure
    for tissue in TISSUES:
        tissue_dir = f"{tissue}_release"
        max_effects[tissue] = {}  # Initialize dictionary for each tissue

        with open(
            f"{working_dir}/{tissue_dir}/connected_component_perturbations_2_hop.pkl",
            "rb",
        ) as f:
            perturbations_stored = pickle.load(f)
            perturbations = perturbations_stored["single"]

        for gene, value in perturbations.items():
            for source in value:
                # Remove tissue identifier from the source name if it exists
                # e.g., 'chr17_40110764_enhancer_adrenal' -> 'chr17_40110764_enhancer'
                base_source = source.rsplit(f"_{tissue}", 1)[0]

                # Initialize or update the maximum effect for this source in this tissue
                if base_source not in max_effects[tissue]:
                    max_effects[tissue][base_source] = perturbations[gene][source][
                        "fold_change"
                    ]
                elif (
                    max_effects[tissue][base_source]
                    < perturbations[gene][source]["fold_change"]
                ):
                    max_effects[tissue][base_source] = perturbations[gene][source][
                        "fold_change"
                    ]


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

    resource_dir = "/Users/steveho/gnn_plots/graph_resources/local"

    set_matplotlib_publication_parameters()
    for node_type in ["enhancer", "promoter", "dyadic", "gene"]:
        if node_type == "enhancer":
            node_file = f"{resource_dir}/enhancer_epimap_screen_overlap.bed"
        elif node_type == "promoter":
            node_file = f"{resource_dir}/promoter_epimap_screen_overlap.bed"
        elif node_type == "dyadic":
            node_file = f"{resource_dir}/dyadic_epimap_screen_overlap.bed"
        else:
            node_file = f"{resource_dir}/gencode_v26_genes_only_with_GTEx_targets.bed"

        # load the node_file
        node_types = pybedtools.BedTool(node_file)
        node_order = [line[3] for line in node_types]

        # Create mapping from enhancer to index
        node_to_idx = {node: idx for idx, node in enumerate(node_order)}

        # Create figure with minimal margins
        plt.figure(figsize=(5, 2.5))

        # Plot each tissue separately to maintain coloring
        for tissue in sorted(max_effects.keys()):
            x_points = []
            y_points = []
            for node, effect in max_effects[tissue].items():
                if (
                    node_type in ["enhancer", "promoter", "dyadic"]
                    and node_type in node
                    or node_type not in ["enhancer", "promoter", "dyadic"]
                    and "ENSG" in node
                ):
                    x_points.append(node_to_idx[node])
                    y_points.append(effect)

            plt.scatter(
                x_points,
                y_points,
                label=tissue,
                alpha=0.65,
                s=4,
                linewidth=0,
                marker=".",
            )

        # Adjust layout to minimize whitespace
        plt.margins(x=0.01)  # Reduce horizontal margins to 1%
        plt.xlim(left=-100)  # Start slightly before 0
        if node_type == "gene":
            plt.ylim(top=5)
            plt.ylim(bottom=-2)
        plt.xlim(right=len(node_order))  # End slightly after max

        node_label = node_type.capitalize()
        plt.xlabel(f"{node_label} Index")
        plt.ylabel("Perturbation Effect")
        plt.title("Max Perturbation Effects by Tissue")
        plt.tight_layout()
        plt.savefig(f"{node_type}_max_effect.png", dpi=450, bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":
    main()

"""In [57]: tissue_medians = {tissue: np.median(list(effects.values()))
...:                  for tissue, effects in max_effects.items()}
...:
...: # If you want to see them sorted from highest to lowest median:
...: sorted_tissues = dict(sorted(tissue_medians.items(),
...:                            key=lambda x: x[1],
...:                            reverse=True))
...:
...: # Print results
...: for tissue, median in sorted_tissues.items():
...:     print(f"{tissue}: {median:.5f}")
...:
aorta: 0.00095
spleen: 0.00094
nhek: 0.00088
k562: 0.00086
gm12878: 0.00084
h1_esc: 0.00084
hepg2: 0.00083
lung: 0.00080
hippocampus: 0.00076
small_intestine: 0.00075
ovary: 0.00075
liver: 0.00075
mammary: 0.00075
skeletal_muscle: 0.00072
left_ventricle: 0.00070
imr90: 0.00070
hmec: 0.00068
pancreas: 0.00061
skin: 0.00057
adrenal: 0.00056

    """
