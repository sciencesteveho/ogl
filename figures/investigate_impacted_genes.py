#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Get overlap between node features and elements within the k-hop subgraph of a
highly predicted gene.
"""


import pickle
import re
import time
from typing import Dict, List, Optional, Tuple, Union

from gseapy import dotplot  # type: ignore
import gseapy as gp  # type: ignore
from matplotlib.colors import LinearSegmentedColormap  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import pybedtools  # type: ignore
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import load_gencode_lookup
from omics_graph_learning.visualization import set_matplotlib_publication_parameters

FEATURES = {
    5: "Size",
    6: "GC-content",
    7: "ATAC",
    8: "CNV",
    9: "CpG methylation",
    10: "CTCF",
    11: "DNase",
    12: "H3K27ac",
    13: "H3K27me3",
    14: "H3K36me3",
    15: "H3K4me1",
    16: "H3K4me2",
    17: "H3K4me3",
    18: "H3K79me2",
    19: "H3K9ac",
    20: "H3K9me3",
    21: "Indels",
    22: "LINE",
    23: "Long terminal repeats",
    24: "Microsatellites",
    25: "PhastCons",
    26: "POLR2A",
    27: "PolyA sites",
    28: "RAD21",
    29: "RBP binding sites",
    30: "Recombination rate",
    31: "Rep G1b",
    32: "Rep G2",
    33: "Rep S1",
    34: "Rep S2",
    35: "Rep S3",
    36: "Rep S4",
    37: "RNA repeat",
    38: "Simple repeats",
    39: "SINE",
    40: "SMC3",
    41: "SNP",
}


def get_khop_from_perturb() -> None:
    """Use the perturbation data to get a list of each k-hop subgraph for each
    highly affected gene.
    """
    pass


def main() -> None:
    """Main function"""
    working_dir = "/Users/steveho/gnn_plots"
    gencode_file = (
        f"{working_dir}/graph_resources/local/gencode_to_genesymbol_lookup_table.txt"
    )

    # load gencode to symbol mapping
    symbol_to_gencode = load_gencode_lookup(gencode_file)
    gencode_to_symbol = invert_symbol_dict(symbol_to_gencode)

    local_dir = "/Users/steveho/gnn_plots/graph_resources/local"
    interpretation_dir = "/Users/steveho/gnn_plots/interpretation/"
    sample = "k562"
    data_dir = f"{interpretation_dir}/{sample}_release"

    enhancers = f"{local_dir}/enhancer_epimap_screen_overlap.bed"
    gencode = f"{local_dir}/gencode_v26_genes_only_with_GTEx_targets.bed"
    promoters = f"{local_dir}/promoter_epimap_screen_overlap.bed"
    dyadic = f"{local_dir}/dyadic_epimap_screen_overlap.bed"

    enhancers = pybedtools.BedTool(enhancers)
    gencode = pybedtools.BedTool(gencode)
    promoters = pybedtools.BedTool(promoters)
    dyadic = pybedtools.BedTool(dyadic)

    # combine all
    elements = (
        enhancers.cat(
            *[gencode, promoters, dyadic], force_truncate=False, postmerge=False
        )
        .sort()
        .saveas()
    )

    # combine all elements

    micro = f"{local_dir}/microsatellites_hg38.bed"
    cpg = "/Users/steveho/gnn_plots/k562_data/cpg_k562_parsed.bed"
    h3k27ac = "/Users/steveho/gnn_plots/k562_data/H3K27ac_k562.bed"
    dnase = "/Users/steveho/gnn_plots/k562_data/DNase_k562.bed"
    atac = "/Users/steveho/gnn_plots/k562_data/ATAC_k562.bed"
    h3k27me3 = "/Users/steveho/gnn_plots/k562_data/H3K27me3_k562.bed"
    h3k4me1 = "/Users/steveho/gnn_plots/k562_data/H3K4me1_k562.bed"
    line = "/Users/steveho/gnn_plots/k562_data/line_hg38.bed"

    # load most impacted genes
    with open(f"{data_dir}/node_feature_top_genes.pkl", "rb") as f:
        most_impacted_genes = pickle.load(f)

    # load 2-hop perturbation
    with open(f"{data_dir}/connected_component_perturbations_2_hop.pkl", "rb") as f:
        perturbations = pickle.load(f)

    # get the single perturbations for full graph
    perturbations = perturbations["single"]

    # exploratory for microsatellites
    # feature = "microsatellites"
    # idx = 24
    # feature_bed = pybedtools.BedTool(micro)
    # feature = "h3k4me1"
    # idx = 15
    # feature_bed = pybedtools.BedTool(h3k4me1)
    # feature = "cpg"
    # idx = 9
    # feature_bed = pybedtools.BedTool(cpg)
    # feature = "h3k27ac"
    # idx = 12
    # feature_bed = pybedtools.BedTool(h3k27ac)

    feat_exps = [
        ("dnase", 11, pybedtools.BedTool(dnase)),
        ("atac", 7, pybedtools.BedTool(atac)),
        ("h3k27me3", 13, pybedtools.BedTool(h3k27me3)),
        ("line", 22, pybedtools.BedTool(line)),
    ]

    for feature, idx, feature_bed in feat_exps:
        ms_genes = most_impacted_genes[idx]
        top = ms_genes[:100]
        # top_genes = [gencode_to_symbol.get(gene.split("_")[0]) for gene, _ in top]

        # filter perturbations for top genes
        # note: some genes may not be in the perturbations, which are filtered for high tpm
        top_perturbations = {
            gene: perturbations[gene] for gene, _ in top if gene in perturbations
        }

        # get a list of k-hop subgraphs
        # first, add key to a list
        # then add each sub_dict key to the list
        subgraphs = [[gene] + list(perturbations[gene]) for gene in top_perturbations]
        fcs = dict(top)

        for idx, subgraph in enumerate(subgraphs):
            k_subgraph = subgraphs[idx]
            gene = k_subgraph[0]

            # remove _k562 from gene name
            k_subgraph = [re.sub(r"_k562", "", gene) for gene in k_subgraph]

            # fc for the gene
            try:
                fc = fcs[gene]
            except KeyError:
                continue

            # get a bedtool object of the subgraph
            subgraph_bed = (
                elements.filter(lambda x: x.name in k_subgraph).sort().saveas()
            )

            # equivalent of sort -u
            subgraph_bed = pybedtools.BedTool(list(set(subgraph_bed))).sort().saveas()

            # get the overlap
            overlap = subgraph_bed.intersect(feature_bed, wa=True, wb=True).sort()

            # print overlap
            print(f"Overlap for subgraph {idx} with {feature}")
            print(f"Gene of interest: {gene}, FC: {fc}")
            print(overlap)

            # write to file
            with open(
                f"/Users/steveho/gnn_plots/interpretation/overlaps/{sample}_{feature}.bed",
                "a",
            ) as f:
                f.write(f"# Overlap for subgraph {idx} with {feature}\n")
                f.write(f"# Gene of interest: {gene}, FC: {fc}\n")
                f.write(str(overlap))


if __name__ == "__main__":
    main()
