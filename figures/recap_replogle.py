#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Scratch code for recapitulating gene perturbations from Replogle et al.

We download DEGs from the day 8 CRISPRi assay from the Replogle et al. paper. We
then filter the table to grab target gene and the number of DEGs via AD test.
"""


import csv
import pickle
from typing import Dict

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore

from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import load_gencode_lookup
from omics_graph_learning.visualization import set_matplotlib_publication_parameters


def filter_perturbations_for_gg(
    perturbations: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Filter nested perturbations dictionary to only include gene-gene
    interactions.
    """
    return {
        perturb: {
            key: value for key, value in perturbations[perturb].items() if "ENSG" in key
        }
        for perturb in perturbations
    }


def _flatten_fold_changes(nested_dict):
    """Flatten nested dictionary of perturbations to a list of tuples."""
    result = []
    for outer_dict in nested_dict.values():
        result.extend(
            [(inner_key, data["fold_change"]) for inner_key, data in outer_dict.items()]
        )
    return result


def main() -> None:
    """Main function"""
    working_dir = "/Users/steveho/gnn_plots"
    gencode_file = (
        f"{working_dir}/graph_resources/local/gencode_to_genesymbol_lookup_table.txt"
    )

    # load gencode to symbol mapping
    symbol_to_gencode = load_gencode_lookup(gencode_file)
    gencode_to_symbol = invert_symbol_dict(symbol_to_gencode)

    # load perturb data
    degs: Dict[str, int] = {}
    with open("repo_degs.txt", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if "ENSG" in line[0]:
                gene = line[0].split("_")[1]
                degs[gene] = int(line[1])

    # load
    with open("k562_release/connected_component_perturbations_2_hop.pkl", "rb") as f:
        perturbations = pickle.load(f)

    # focused on single
    perturbations = perturbations["single"]

    # filter to get the gene subkeys
    perturbations = filter_perturbations_for_gg(perturbations)

    # get perturbation genes
    perturb_genes = set()
    for perturb in perturbations:
        for gene in perturbations[perturb]:
            perturb_genes.add(gene.split("_")[0])

    perturb_genes_symbols = [gencode_to_symbol.get(gene) for gene in perturb_genes]

    # filter degs - 7843 genes left
    degs = {gene: degs[gene] for gene in degs if gene in perturb_genes_symbols}

    # get median degs
    median_degs = np.median(list(degs.values()))  # 2.0

    # sort into > 10 deg and < 10 deg
    degs_q1 = [gene for gene in degs if degs[gene] < median_degs]
    degs_q2 = [gene for gene in degs if degs[gene] == median_degs]

    # get a new dictionary mapping the average impact via deletion of the gene in each quartile
    # first, get the fold changes as a list of tuples instead
    perturb_values = _flatten_fold_changes(perturbations)

    # rename the keys to symbols
    perturb_values = [
        (gencode_to_symbol.get(gene.split("_")[0]), fc) for gene, fc in perturb_values
    ]

    # loop through perturb values
    # if the gene is in degs_q1, degs_q2, degs_q3, degs_q4, add to an array to plot the fold changes
    degs_q1_fc = []
    degs_q2_fc = []
    for gene, fc in perturb_values:
        if gene in degs_q1:
            degs_q1_fc.append(fc)
        elif gene in degs_q2:
            degs_q2_fc.append(fc)

    # plot the fold changes in a boxplot
    set_matplotlib_publication_parameters()
    fig, ax = plt.subplots(figsize=(6, 4))
    parts = ax.violinplot([degs_q1_fc, degs_q2_fc], bw=0.1, inner="quartile")

    ax.set_ylabel("Fold Change")
    ax.set_title("Distribution of Fold Changes by Quartile")

    plt.tight_layout()
    plt.savefig("replogle_perturbations.png", dpi=450)
    plt.clf()


if __name__ == "__main__":
    main()
