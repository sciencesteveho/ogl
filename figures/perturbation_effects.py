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
import matplotlib.ticker as ticker  # type: ignore
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
    attn_mapped: Dict[Tuple[str, str], float], num_connections: int = 100
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


def load_single_perturbation_effects(
    tissue: str, working_dir: str = "/Users/steveho/gnn_plots/interpretation"
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load the systematic single perturbations."""
    tissue_dir = f"{tissue}_release"
    with open(
        f"{working_dir}/{tissue_dir}/connected_component_perturbations_2_hop.pkl", "rb"
    ) as f:
        perturbations_stored = pickle.load(f)
        perturbations = perturbations_stored["single"]

    return perturbations


def get_noderefs(resource_dir: str) -> Dict[str, str]:
    """Return the appropriate node reference file for each node type."""
    return {
        "enhancer": f"{resource_dir}/enhancer_epimap_screen_overlap.bed",
        "promoter": f"{resource_dir}/promoter_epimap_screen_overlap.bed",
        "dyadic": f"{resource_dir}/dyadic_epimap_screen_overlap.bed",
        "gene": f"{resource_dir}/gencode_v26_genes_only_with_GTEx_targets.bed",
    }


def get_node_type(element: str) -> str:
    """Classify a node into one of four types based on string matching."""
    if "enhancer" in element:
        return "enhancer"
    elif "promoter" in element:
        return "promoter"
    elif "dyadic" in element:
        return "dyadic"
    else:
        return "gene"


def process_tissue(
    tissue: str,
) -> Tuple[
    str,
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, Dict[str, float]]],
]:
    """Process a single tissue by:
    1. Loading its perturbation effects
    2. Aggregating fold changes for each element (max and avg)
    3. Sorting and collecting the top 25 fold changes for each node type
    """
    perturbations = load_single_perturbation_effects(tissue=tissue)

    store_effects: Dict[str, Dict[str, List[float]]] = {}
    node_entries: Dict[str, List[Tuple[float, str, str]]] = {
        "enhancer": [],
        "promoter": [],
        "dyadic": [],
        "gene": [],
    }

    # aggregate fold changes for each element
    for gene, gene_values in perturbations.items():
        for source, data in gene_values.items():
            element = source.rsplit(f"_{tissue}", 1)[0]
            fold_change = data["fold_change"]

            store_effects.setdefault(gene, {}).setdefault(element, []).append(
                fold_change
            )

            # accumulate for later sorting
            node_type = get_node_type(element)
            node_entries[node_type].append((fold_change, gene, element))

    # get all fold changes for each element
    element_fc_map: Dict[str, List[Tuple[str, float]]] = {}
    for gene, elem_vals in store_effects.items():
        for element, fc_list in elem_vals.items():
            for fc in fc_list:
                element_fc_map.setdefault(element, []).append((gene, fc))

    # build max and avg dictionaries
    max_perturbations: Dict[str, Dict[str, str]] = {}
    average_perturbations: Dict[str, Dict[str, str]] = {}

    for element, gene_fc_list in element_fc_map.items():
        # get max
        best_gene, max_val = max(gene_fc_list, key=lambda x: x[1])

        # compute average
        all_fcs = [fc for (_, fc) in gene_fc_list]
        avg_val = sum(all_fcs) / len(all_fcs)

        # if there's only one gene, use that gene's name; else store "multiple"
        unique_genes = {g for (g, _) in gene_fc_list}
        if len(unique_genes) == 1:
            (sole_gene,) = unique_genes
            avg_gene_name = sole_gene
        else:
            avg_gene_name = "multiple"

        max_perturbations[element] = {
            "fc": str(max_val),
            "gene": best_gene,
        }
        average_perturbations[element] = {
            "fc": str(avg_val),
            "gene": avg_gene_name,
        }

    top_subgraphs: Dict[str, Dict[str, Dict[str, float]]] = {
        "enhancer": {},
        "promoter": {},
        "dyadic": {},
        "gene": {},
    }

    for nodetype, entries in node_entries.items():
        # sort descending by absolute fc
        entries.sort(key=lambda x: abs(x[0]), reverse=True)

        top_distinct = []
        seen_elements = set()

        for fc, gene, element in entries:
            if element not in seen_elements:
                top_distinct.append((fc, gene, element))
                seen_elements.add(element)
            if len(top_distinct) >= 25:
                break

        # insert into the nested dictionaries
        for fc, gene, element in top_distinct:
            # safest way for mypy
            if gene not in top_subgraphs[nodetype]:
                top_subgraphs[nodetype][gene] = {}
            top_subgraphs[nodetype][gene][element] = fc

    return tissue, max_perturbations, average_perturbations, top_subgraphs


def collate_perturbation_effect() -> Tuple[
    Dict[str, Dict[str, Dict[str, str]]],
    Dict[str, Dict[str, Dict[str, str]]],
    Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
]:
    """Collate perturbation effects across all tissues by computing both the
    maximum and average fold changes, plus a nested dictionary describing the
    top 25 largest fc for distinct elements.
    """
    max_effects: Dict[str, Dict[str, Dict[str, str]]] = {}
    avg_effects: Dict[str, Dict[str, Dict[str, str]]] = {}
    top_subgraphs: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for t in tqdm(TISSUES, desc="Processing tissues"):
        tissue, tissue_max, tissue_avg, tissue_top = process_tissue(t)
        max_effects[tissue] = tissue_max
        avg_effects[tissue] = tissue_avg
        top_subgraphs[tissue] = tissue_top

    return max_effects, avg_effects, top_subgraphs


def _get_noderef_bedtool(element: str, noderefs: Dict[str, str]) -> List[str]:
    """Return a BedTool object for a given element type and its corresponding
    node order.
    """
    node_file = noderefs[element]
    return [line[3] for line in BedTool(node_file)]


def _filter_perturbation_effects(
    effects: Dict[str, Dict[str, float]],
    threshold: float = 0.0144,
) -> Dict[str, Dict[str, float]]:
    """Filter out perturbation effects below a certain threshold."""
    return {
        tissue: {
            element: effect
            for element, effect in data.items()
            if abs(effect) > threshold
        }
        for tissue, data in effects.items()
    }


def plot_global_perturbation_effects(
    effects: Dict[str, Dict[str, float]],
    element: str,
    noderefs: Dict[str, str],
    effect_mode: str,
    filter: bool = True,
) -> None:
    """Plot the perturbation effects.
    X axis: element index, sorted by linear genomic coordinate
    Y axis: perturbation effect
    Cbar: tissue

    We add a boolean flag to filter out lower effect elements for better
    visualization. We filter out elements with less than a 1% predicted fold
    change, which is the average effect size for K562 crispri cCREs. Thus is
    equivalent to a 0.0144 fold change in the log2 space.
    """
    n_features = len(TISSUES)
    feature_colors = plt.cm.tab10(np.linspace(0, 1, n_features))  # type: ignore

    # get node order for x axis
    node_order = _get_noderef_bedtool(element, noderefs)

    # mapping from enhancer to index
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}

    # filter out lower effect elements
    if filter:
        effects = _filter_perturbation_effects(effects)

    # create figure with minimal margins
    plt.figure(figsize=(5, 2.25))

    # store all points to avoid dominance by last tissue
    all_x = []
    all_y = []
    all_colors = []
    element_count = 0

    # plot each tissue for global view
    # color each tissue differently
    for tissue in sorted(TISSUES):
        x_points = []
        y_points = []
        for node, effect in effects[tissue].items():
            if (
                element in {"enhancer", "promoter", "dyadic"}
                and element in node
                or element not in ["enhancer", "promoter", "dyadic"]
                and "ENSG" in node
            ):
                x_points.append(node_to_idx[node])
                y_points.append(effect)

        all_x.extend(x_points)
        all_y.extend(y_points)
        all_colors.extend(
            [feature_colors[list(sorted(TISSUES)).index(tissue)]] * len(x_points)
        )
        element_count += len(x_points)

    # randomize order of points to avoid color dominance
    indices = np.random.permutation(len(all_x))
    all_x = np.array(all_x)[indices]
    all_y = np.array(all_y)[indices]
    all_colors = np.array(all_colors)[indices]

    plt.scatter(all_x, all_y, alpha=0.75, s=4, linewidth=0, marker=".", c=all_colors)

    # adjust layout to minimize whitespace
    plt.margins(x=0, y=0.01)
    if element == "gene":
        plt.ylim(top=5)
        plt.ylim(bottom=-1.35)

    # set labels
    plt.ylabel(r"Log$_2$ fold change")
    plt.title(f"Global view of {element} perturbation effects ({effect_mode})")
    plt.xticks([])

    # add x-axis label with line annotation
    fig = plt.gcf()
    ax = plt.gca()

    # remove the current xlabel
    ax.set_xlabel("")

    # add text label aligned to the left
    element_label = f"{element}s" if element != "dyadic" else element

    text = ax.text(
        0.01,
        -0.085,
        f"{element_count:,} {element_label}",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
    )

    # add arrow
    ax.annotate(
        "",
        xy=(0.3, -0.045),
        xytext=(0, -0.045),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color="black", linewidth=0.5, shrinkA=0, shrinkB=0
        ),
    )

    plt.tight_layout()
    plt.savefig(
        f"{element}_{effect_mode}_effect_global.png", dpi=450, bbox_inches="tight"
    )
    plt.close()
    plt.clf()


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


def plot_sample_perturbation_effects(
    tissue: str,
    tissue_name: str,
    effects_tissue: Dict[str, Dict[str, str]],
    gencode_coordinates: Dict[str, Tuple[str, int, int]],
    element_coordinates: Dict[str, Tuple[str, int, int]],
    element: str,
) -> None:
    """Plot the perturbation effects.
    X axis: element index, sorted by linear genomic coordinate
    Y axis: linear genomic distance to gene
    Cbar: effect strength

    We use the same boolean flag as in plot_global to filter out lower effect
    elements.
    """
    distances = build_assigned_gene_distances(
        effects_for_tissue=effects_tissue,
        gene_coords=gencode_coordinates,
        element_coords=element_coordinates,
    )
    sorted_elements = sort_elements_by_coordinate(element_coordinates)

    x_vals = []
    y_vals = []
    c_vals: List[float] = []

    for i, elem_id in enumerate(sorted_elements):
        if elem_id not in effects_tissue:
            continue

        if elem_id not in distances:
            continue

        fc = effects_tissue[elem_id].get("fc", None)
        if not fc:
            continue

        try:
            fc_val = float(fc)
        except ValueError:
            continue

        # filter out small effects
        if abs(fc_val) < 0.0144:
            continue

        # get the distance to the assigned gene in kb
        dist_bp = distances[elem_id]
        dist_kb = dist_bp / 1000.0

        x_vals.append(i)
        y_vals.append(dist_kb)
        c_vals.append(fc_val)

    if not x_vals:
        print(f"No data to plot for {tissue}")
        return

    x_values = np.array(x_vals)
    y_values = np.array(y_vals)
    c_values = np.array(c_vals)

    # sort points to plot higher fc values later so they dont drown
    sort_idx = np.argsort(c_values)
    x_values = x_values[sort_idx]
    y_values = y_values[sort_idx]
    c_values = c_values[sort_idx]

    plt.figure(figsize=(5, 2.25))
    plt.margins(x=0, y=0.01)
    plt.title(f"Global view of {element} perturbation effects in {tissue_name}")

    min_val = min(c_values)
    max_val = max(c_values)
    max_abs_val = max(abs(min_val), abs(max_val))

    # custom normalization class to enhance color contrast
    class EnhancedNorm(mcolors.Normalize):
        def __init__(self, vmin=None, vmax=None, enhance_factor=0.3):
            self.enhance_factor = enhance_factor
            super().__init__(vmin, vmax)

        def __call__(self, value, clip=None):
            # sigmoidal enhancement of mid-range values
            result = np.ma.masked_array(value)
            if not self.vmin == self.vmax:
                # scale to [-1, 1] range first
                result = 2 * (result - self.vmin) / (self.vmax - self.vmin) - 1
                # apply enhancement that strengthens mid-range values
                result = np.sign(result) * (np.abs(result) ** self.enhance_factor)
                # scale back to [0, 1] for colormap
                result = (result + 1) / 2
            return result

    norm = EnhancedNorm(vmin=-max_abs_val, vmax=max_abs_val, enhance_factor=0.7)

    sc = plt.scatter(
        x_values,
        y_values,
        c=c_values,
        cmap="RdBu_r",
        s=3,
        alpha=0.8,
        linewidths=0,
        norm=norm,
    )

    ticks = [-max_abs_val, 0, max_abs_val]
    cbar = plt.colorbar(sc, shrink=0.35, aspect=5, ticks=ticks)

    # Format tick labels nicely
    tick_labels = [f"{t:.1f}" for t in ticks]
    cbar.ax.set_yticklabels(tick_labels)

    cbar.set_label(r"Log$_2$ fold change")
    cbar.ax.tick_params(labelsize=8)

    plt.ylabel("Distance to assigned gene (kb)")
    plt.xticks([])
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    fig = plt.gcf()
    ax = plt.gca()

    # remove the current xlabel
    ax.set_xlabel("")

    # add text label aligned to the left
    element_label = f"{element}s" if element != "dyadic" else element

    text = ax.text(
        0.01,
        -0.085,
        f"{len(x_values):,} {element_label}",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
    )

    # add arrow
    ax.annotate(
        "",
        xy=(0.3, -0.045),
        xytext=(0, -0.045),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color="black", linewidth=0.5, shrinkA=0, shrinkB=0
        ),
    )

    plt.tight_layout()
    out_name = f"{tissue}_{element}.png"
    plt.savefig(out_name, dpi=450)
    plt.close()


def main() -> None:
    """Main function."""
    set_matplotlib_publication_parameters()
    resource_dir = "/Users/steveho/gnn_plots/graph_resources/local"

    # get reference files for each node type
    noderefs = get_noderefs(resource_dir)

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
        node_1 = idxs[idx_1].strip("_k562")
        node_2 = idxs[idx_2].strip("_k562")
        attn_mapped[(node_1, node_2)] = val

    # get top gene connections
    top_attn_connections = get_top_attn_connections_to_genes(
        attn_mapped, num_connections=100
    )

    max_effects, avg_effects, top_perturbations = collate_perturbation_effect()

    # save results for later
    with open("max_effects.pkl", "wb") as f:
        pickle.dump(max_effects, f)

    with open("avg_effects.pkl", "wb") as f:
        pickle.dump(avg_effects, f)

    with open("top_perturbations.pkl", "wb") as f:
        pickle.dump(top_perturbations, f)

    # plot global perturbation effects
    for effects in [max_effects, avg_effects]:
        effect_mode = "max" if effects == max_effects else "average"
        print(f"Plotting global perturbation effects for {effect_mode} effects")
        for element in ["enhancer", "promoter", "dyadic", "gene"]:
            flattened_effects = {
                tissue: {
                    element_idx: float(data["fc"])
                    for element_idx, data in effects[tissue].items()
                }
                for tissue in effects
            }

            plot_global_perturbation_effects(
                effects=flattened_effects,
                element=element,
                noderefs=noderefs,
                filter=True,
                effect_mode=effect_mode,
            )

    # plot individual perturbation effects
    gencode_coordinates = load_gene_coordinates(gencode)
    for tissue in TISSUES:
        for element in ["enhancer", "promoter", "dyadic", "gene"]:
            # get coordinates for the element
            element_coordinates = load_element_coordinates(BedTool(noderefs[element]))

            plot_sample_perturbation_effects(
                tissue=tissue,
                tissue_name=TISSUES[tissue],
                effects_tissue=max_effects[tissue],
                gencode_coordinates=gencode_coordinates,
                element_coordinates=element_coordinates,
                element=element,
            )


if __name__ == "__main__":
    main()
