#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code for figure 3.
"""


from collections import defaultdict
import csv
import multiprocessing as mp
from pathlib import Path
import pickle
import random
from typing import Dict, List, Optional, Tuple, Union

import matplotlib  # type: ignore
from matplotlib.collections import PolyCollection  # type: ignore
from matplotlib.colors import ListedColormap  # type: ignore
from matplotlib.colors import TwoSlopeNorm  # type: ignore
import matplotlib.colors as mcolors
from matplotlib.figure import Figure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from numpy import sqrt
import numpy as np
import pandas as pd
from pybedtools import BedTool  # type: ignore
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
import torch
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.interpret_utils import get_gene_idx_mapping
from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import load_gencode_lookup
from omics_graph_learning.interpret.interpret_utils import map_symbol
from omics_graph_learning.utils.common import get_physical_cores
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


def get_top_attn_connections_to_genes(
    attn_mapped: Dict[Tuple[str, str], float], num_connections: int = 10
) -> Dict[str, Dict[str, float]]:
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

        if len(values) <= num_connections:
            top_indices = np.argsort(values)[::-1]
        else:
            top_indices = np.argpartition(values, -num_connections)[-num_connections:]
            top_indices = top_indices[np.argsort(values[top_indices])][::-1]

        result[gene_target] = {sources[i]: values[i] for i in top_indices}

    return result


def load_gene_coordinates(gencode_bed: BedTool) -> Dict[str, Tuple[str, int, int]]:
    """Convert gencode bedtool into dict structure."""
    return {
        feature.name: (feature.chrom, int(feature.start), int(feature.end))
        for feature in gencode_bed
    }


def load_element_coordinates(element_bed: BedTool) -> Dict[str, Tuple[str, int, int]]:
    """Convert element bedtool into dict structure."""
    return {
        feature.name: (feature.chrom, int(feature.start), int(feature.end))
        for feature in element_bed
    }


def build_assigned_gene_distances(
    effects_for_tissue: Dict[str, dict[str, str]],
    gene_coords: Dict[str, tuple[str, int, int]],
    element_coords: Dict[str, tuple[str, int, int]],
) -> Dict[str, int]:
    """For each element in effects_for_tissue, compute the
    signed distance to *its assigned gene* from 'gene_coords'.

    If an element or gene is missing from coords, skip it.
    """
    distances = {}
    for element_id, data in effects_for_tissue.items():
        gene_id = data.get("gene")
        if not gene_id:
            continue

        # remove tissue identifier
        gene_id = gene_id.split("_")[0]

        # ensure we have coords
        if gene_id not in gene_coords:
            continue
        if element_id not in element_coords:
            continue

        try:
            dist_bp = compute_signed_distance(
                gene_coords[gene_id], element_coords[element_id]
            )
        except ValueError:
            print(f"Error computing distance for {element_id} and {gene_id}")

        distances[element_id] = dist_bp
    return distances


def compute_signed_distance(
    gene_info: tuple[str, int, int],
    elem_info: tuple[str, int, int],
) -> int:
    """Signed distance from element midpoint to gene body midpoint."""
    g_chrom, g_start, g_end = gene_info
    e_chrom, e_start, e_end = elem_info

    # ensure same chromosome
    if g_chrom != e_chrom:
        print(f"Chromosomes do not match: {g_chrom} vs {e_chrom}")
        print(f"Gene: {g_chrom}:{g_start}-{g_end}")
        print(f"Element: {e_chrom}:{e_start}-{e_end}")
        raise ValueError("Chromosomes do not match")

    anchor = (g_start + g_end) // 2

    elem_mid = (e_start + e_end) // 2
    return elem_mid - anchor


def sort_elements_by_coordinate(
    element_coords: Dict[str, Tuple[str, int, int]]
) -> List[str]:
    """Returns a list of element_ids sorted by (chrom, start)."""
    items = list(element_coords.items())

    def chromosome_to_int(chrom: str) -> int:
        """Convert chromosome string to integer for sorting."""
        chrom = chrom.removeprefix("chr")
        return int(chrom) if chrom.isdigit() else ord(chrom[0])

    items.sort(key=lambda x: (chromosome_to_int(x[1][0]), x[1][1]))
    return [elem_id for (elem_id, _) in items]


def main() -> None:
    """Main function."""
    set_matplotlib_publication_parameters()
    resource_dir = "/Users/steveho/gnn_plots/graph_resources/local"

    # load gencode for gene coordinates
    gencode_file = "/Users/steveho/gnn_plots/graph_resources/local/gencode_v26_genes_only_with_GTEx_targets.bed"
    gencode = BedTool(gencode_file)

    # load attention weights
    attn = "/Users/steveho/gnn_plots/interpretation/k562_release/attention_weights_genes.pt"
    attn_weights = torch.load(attn)

    # load indexes
    index_file = (
        "/Users/steveho/gnn_plots/graph_resources/idxs/k562_release_full_graph_idxs.pkl"
    )
    idxs = pickle.load(open(index_file, "rb"))
    idxs = {v: k for k, v in idxs.items()}  # reverse map

    # map idx tuples in attn_weights back to genes and stuff
    attn_mapped = {}
    for (idx_1, idx_2), val in attn_weights.items():
        node_1 = (
            idxs[idx_1].removesuffix("_k562")
            if idxs[idx_1].endswith("_k562")
            else idxs[idx_1]
        )
        node_2 = (
            idxs[idx_2].removesuffix("_k562")
            if idxs[idx_2].endswith("_k562")
            else idxs[idx_2]
        )
        attn_mapped[(node_1, node_2)] = val

    # filter for greater than median
    median_attn = np.median(list(attn_mapped.values()))
    strong_attn = {k: v for k, v in attn_mapped.items() if v > median_attn}

    # get top gene connections
    top_attn_connections = get_top_attn_connections_to_genes(
        attn_mapped, num_connections=1000
    )


if __name__ == "__main__":
    main()
