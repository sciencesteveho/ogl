#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for genome data processing"""

import csv
from datetime import timedelta
import functools
import inspect
import os
import pickle
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from cmapPy.pandasGEXpress.parse_gct import parse
import networkx as nx
import numpy as np
import pandas as pd
import pybedtools
import yaml

ATTRIBUTES = [
    "gc",
    "atac",
    "cnv",
    "cpg",
    "ctcf",
    "dnase",
    "h3k27ac",
    "h3k27me3",
    "h3k36me3",
    "h3k4me1",
    "h3k4me2",
    "h3k4me3",
    "h3k79me2",
    "h3k9ac",
    "h3k9me3",
    "indels",
    "line",
    "ltr",
    "microsatellites",
    "phastcons",
    "polr2a",
    "polyasites",
    "rad21",
    "rbpbindingsites",
    "recombination",
    "repg1b",
    "repg2",
    "reps1",
    "reps2",
    "reps3",
    "reps4",
    "rnarepeat",
    "simplerepeats",
    "sine",
    "smc3",
    "snp",
]

NODES = [
    "cpgislands",
    "crms",
    "ctcfccre",
    "dyadic",
    "enhancers",
    "gencode",
    "promoters",
    "superenhancers",
    "tads",
    "tfbindingsites",
    "tss",
]

NODE_FEAT_IDXS = (
    {
        "start": 0,
        "end": 1,
        "size": 2,
        "gc": 3,
        "atac": 4,
        "cnv": 5,
        "cpg": 6,
        "ctcf": 7,
        "dnase": 8,
        "h3k27ac": 9,
        "h3k27me3": 10,
        "h3k36me3": 11,
        "h3k4me1": 12,
        "h3k4me2": 13,
        "h3k4me3": 14,
        "h3k79me2": 15,
        "h3k9ac": 16,
        "h3k9me3": 17,
        "indels": 18,
        "line": 19,
        "ltr": 20,
        "microsatellites": 21,
        "phastcons": 22,
        "polr2a": 23,
        "polyasites": 24,
        "rad21": 25,
        "rbpbindingsites": 26,
        "recombination": 27,
        "repg1b": 28,
        "repg2": 29,
        "reps1": 30,
        "reps2": 31,
        "reps3": 32,
        "reps4": 33,
        "rnarepeat": 34,
        "simplerepeats": 35,
        "sine": 36,
        "smc3": 37,
        "snp": 38,
    },
)

TISSUES = [
    "hippocampus",
    "left_ventricle",
    "liver",
    "lung",
    "mammary",
    "pancreas",
    "skeletal_muscle",
    "skin",
    "small_intestine",
    # "hela",
    # "k562",
    # "npc",
]

ONEHOT_EDGETYPE = {
    "g_e": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "gene-enhancer"
    "g_p": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "gene-promoter"
    "g_d": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "gene-dyadic"
    "g_se": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "gene-superenhancer"
    "p_e": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # "promoter-enhancer"
    "p_d": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # "promoter-dyadic"
    "p_se": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # "promoter-superenhancer"
    "g_g": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # "gene-gene"
    "ppi": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # "protein-protein"
    "mirna": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # "mirna-gene"
    "tf_marker": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # "tf-marker"
    "circuits": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # "circuits"
    "local": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # "local"
}

TISSUE_TPM_KEYS = {
    "Adipose - Subcutaneous": 3,
    "Adipose - Visceral (Omentum)": 4,
    "Adrenal Gland": 5,
    "Artery - Aorta": 6,
    "Artery - Coronary": 7,
    "Artery - Tibial": 8,
    "Bladder": 9,
    "Brain - Amygdala": 10,
    "Brain - Anterior cingulate cortex (BA24)": 11,
    "Brain - Caudate (basal ganglia)": 12,
    "Brain - Cerebellar Hemisphere": 13,
    "Brain - Cerebellum": 14,
    "Brain - Cortex": 15,
    "Brain - Frontal Cortex (BA9)": 16,
    "Brain - Hippocampus": 17,
    "Brain - Hypothalamus": 18,
    "Brain - Nucleus accumbens (basal ganglia)": 19,
    "Brain - Putamen (basal ganglia)": 20,
    "Brain - Spinal cord (cervical c-1)": 21,
    "Brain - Substantia nigra": 22,
    "Breast - Mammary Tissue": 23,
    "Cells - Cultured fibroblasts": 24,
    "Cells - EBV-transformed lymphocytes": 25,
    "Cervix - Ectocervix": 26,
    "Cervix - Endocervix": 27,
    "Colon - Sigmoid": 28,
    "Colon - Transverse": 29,
    "Esophagus - Gastroesophageal Junction": 30,
    "Esophagus - Mucosa": 31,
    "Esophagus - Muscularis": 32,
    "Fallopian Tube": 33,
    "Heart - Atrial Appendage": 34,
    "Heart - Left Ventricle": 35,
    "Kidney - Cortex": 36,
    "Kidney - Medulla": 37,
    "Liver": 38,
    "Lung": 39,
    "Minor Salivary Gland": 40,
    "Muscle - Skeletal": 41,
    "Nerve - Tibial": 42,
    "Ovary": 43,
    "Pancreas": 44,
    "Pituitary": 45,
    "Prostate": 46,
    "Skin - Not Sun Exposed (Suprapubic)": 47,
    "Skin - Sun Exposed (Lower leg)": 48,
    "Small Intestine - Terminal Ileum": 49,
    "Spleen": 50,
    "Stomach": 51,
    "Testis": 52,
    "Thyroid": 53,
    "Uterus": 54,
    "Vagina": 55,
    "Whole Blood": 56,
}


def chunk_genes(
    genes: List[str],
    chunks: int,
) -> Dict[int, List[str]]:
    """Constructs graphs in parallel"""
    ### get list of all gencode V26 genes
    for num in range(0, 5):
        random.shuffle(genes)

    split_list = lambda l, chunks: [l[n : n + chunks] for n in range(0, len(l), chunks)]
    split_genes = split_list(genes, chunks)
    return {index: gene_list for index, gene_list in enumerate(split_genes)}


def dir_check_make(dir: str) -> None:
    """Utility to make directories only if they do not already exist"""
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass


def _listdir_isfile_wrapper(dir: str) -> List[str]:
    """Returns a list of files within the directory"""
    return [file for file in os.listdir(dir) if os.path.isfile(f"{dir}/{file}")]


def filtered_genes_from_bed(tpm_filtered_genes: str) -> List[str]:
    with open(tpm_filtered_genes, newline="") as file:
        return [line[3] for line in csv.reader(file, delimiter="\t")]


def gene_list_from_graphs(root_dir: str, tissue: str) -> List[str]:
    """Returns a list of genes with constructed graphs, avoiding genes that may
    not have edges in smaller window"""
    directory = f"{root_dir}/{tissue}/parsing/graphs"
    return [gene.split("_")[0] for gene in os.listdir(directory)]


def genes_from_gff(gff: str) -> List[str]:
    """Get list of gtex genes from GFF file"""
    with open(gff, newline="") as file:
        return {
            line[3]: line[0]
            for line in csv.reader(file, delimiter="\t")
            if line[0] not in ["chrX", "chrY", "chrM"]
        }


def genes_from_gencode(gencode_ref) -> Dict[str, str]:
    """Returns a dict of gencode v26 genes, their ids and associated gene
    symbols
    """
    return {
        line[9].split(";")[3].split('"')[1]: line[3]
        for line in gencode_ref
        if line[0] not in ["chrX", "chrY", "chrM"]
    }


def parse_yaml(config_file: str) -> Dict[str, Union[str, list]]:
    """Load yaml for parsing"""
    with open(config_file, "r") as stream:
        params = yaml.safe_load(stream)
    return params


def time_decorator(print_args: bool = False, display_arg: str = "") -> Callable:
    """Decorator to time functions

    Args:
        print_args (bool, optional): Defaults to False.
        display_arg (str, optional): Decides wether or not args are printed to
        stdout. Defaults to "".
    """

    def _time_decorator_func(function: Callable) -> Callable:
        @functools.wraps(function)
        def _execute(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            fxn_args = inspect.signature(function).bind(*args, **kwargs).arguments
            try:
                result = function(*args, **kwargs)
                return result
            except Exception as error:
                result = str(error)
                raise
            finally:
                end_time = time.monotonic()
                if print_args == True:
                    print(
                        f"Finished {function.__name__} {[val for val in fxn_args.values()]} - Time: {timedelta(seconds=end_time - start_time)}"
                    )
                else:
                    print(
                        f"Finished {function.__name__} {display_arg} - Time: {timedelta(seconds=end_time - start_time)}"
                    )

        return _execute

    return _time_decorator_func


@time_decorator(print_args=True)
def _filter_low_tpm(
    tissue: str,
    file: str,
    return_list: False,
) -> List[str]:
    """
    Filter genes according to the following criteria: (A) Only keep genes
    expressing >= 1 TPM across 20% of samples in that tissue
    """
    df = pd.read_table(file, index_col=0, header=[2])
    sample_n = len(df.columns)
    df["total"] = df.select_dtypes(np.number).ge(0.5).sum(axis=1)
    df["result"] = df["total"] >= (0.20 * sample_n)
    if return_list == False:
        return [f"{gene}_{tissue}" for gene in list(df.loc[df["result"] == True].index)]
    else:
        return list(df.loc[df["result"] == True].index)


@time_decorator(print_args=True)
def _filter_low_tpm_across_tissues(
    file: str,
    tissue: str,
    return_list: False,
    tpm: int = 0.5,
    tissues: List[str] = [
        "Brain - Hippocampus",
        "Breast - Mammary Tissue",
        "Heart - Left Ventricle",
        "Liver",
        "Lung",
        "Pancreas",
        "Muscle - Skeletal",
        "Skin - Not Sun Exposed (Suprapubic)",
        "Small Intestine - Terminal Ileum",
    ],
) -> List[str]:
    """
    Filter genes according to the following criteria: (B) Only keep genes that
    express >= TPM in one of the samples in our tissues
    """
    df = parse(file).data_df
    df = df[tissues]
    if return_list == False:
        return [
            f"{gene}_{tissue}" for gene in list(df.loc[df[df >= tpm].any(axis=1)].index)
        ]
    else:
        return [gene for gene in list(df.loc[df[df >= tpm].any(axis=1)].index)]


@time_decorator(print_args=True)
def _tpm_filter_gene_windows(
    gencode: str,
    tissue: str,
    tpm_file: str,
    slop: bool,
    chromfile: Optional[str] = None,
    window: Optional[int] = 0,
) -> Tuple[pybedtools.BedTool, List[str]]:
    """
    Filter out genes in a GTEx tissue with less than 1 tpm across 20% of
    samples in that tissue. Additionally, we exclude analysis of sex
    chromosomes.

    Returns:
        pybedtools object with +/- <window> windows around that gene
    """
    tpm_filtered_genes = _filter_low_tpm_across_tissues(
        file=tpm_file,
        tissue=tissue,
        return_list=True,
    )
    genes = pybedtools.BedTool(gencode)
    genes_filtered = genes.filter(
        lambda x: x[3] in tpm_filtered_genes and x[0] not in ["chrX", "chrY", "chrM"]
    ).saveas()

    if slop:
        return (
            genes_filtered.sort(),
            genes_filtered.slop(g=chromfile, b=window).cut([0, 1, 2, 3]).sort(),
        )
    else:
        return genes_filtered.sort()
        # [x[3] for x in genes_filtered]


def _get_sorted_node_degrees(graph: nx.Graph) -> None:
    """_summary_

    Args:
        graph (nx.Graph): _description_
    """
    return sorted(graph.degree, key=lambda x: x[1], reverse=True)


def _calculate_max_distance_base_graph(bed: List[List[str]]) -> List[List[str]]:
    """Calculate the max distance between nodes in the base graph. Report the
    max, mean, and median distances for all interaction type data.
    """
    return {
        max(int(line[2]), int(line[6])) - min(int(line[3]), int(line[7]))
        for line in bed
    }


def _graph_stats(tissue, graph_dir):
    """_summary_

    Args:
        tissue (str): _description_
    """
    with open(f"{graph_dir}/graph.pkl", "rb") as file:
        graph = pickle.load(file)

    print(graph["num_nodes"])
    print(graph["num_edges"])
    print(graph["avg_edges"])


def _concat_nx_graphs(tissue_list, graph_dir, graph_type):
    """_summary_

    Args:
        tissue_list (str): _description_
    """
    graph_list = []
    for tissue in tissue_list:
        graph_list.append(
            nx.read_gml(f"{graph_dir}/{tissue}/{tissue}_{graph_type}_graph.gml")
        )

    return nx.compose_all(graph_list)


def _combined_graph_arrays(
    tissue_list: List[str],
    graph_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine graph arrays from multiple tissues"""
    feats = []
    for tissue in tissue_list:
        graph_file = f"{graph_dir}/graph.pkl"
        with open(graph_file, "rb") as f:
            graph = pickle.load(f)
        feats.append(graph["node_feat"])


def _map_genesymbol_to_tss(tss_path: str, annotation_path: str) -> List[str]:
    """_summary_

    Args:
        tss_path (str): _description_
        annotation_path (str): _description_

    Returns:
        List[str]: _description_
    """
    genesymbol_dict = {
        line[0]: line[7] for line in csv.reader(open(annotation_path), delimiter="\t")
    }

    return [
        (line[0], line[1], line[2], f"tss_{line[3]}_{genesymbol_dict[line[3]]}")
        for line in csv.reader(open(tss_path), delimiter="\t")
    ]


@time_decorator(print_args=True)
def _n_ego_graph(
    graph: nx.Graph,
    max_nodes: int,
    node: str,
    radius: int,
) -> nx.Graph:
    """Get n-ego graph centered around a gene (node)"""
    # get n-ego graph
    n_ego_graph = nx.ego_graph(
        graph=graph,
        n=node,
        radius=radius,
        undirected=True,
    )

    # if n_ego_graph is too big, reduce radius until n_ego_graph has nodes < max_nodes
    while n_ego_graph.number_of_nodes() > max_nodes:
        radius -= 1
        n_ego_graph = nx.ego_graph(
            graph=graph,
            n=node,
            radius=radius,
        )

    return n_ego_graph


def filter_target_split(
    root_dir,
    tissues,
    targets,
):
    """Filters and only keeps targets that pass the TPM filter of >1 TPM across
    20% of samples

    Args:
        tissues (Dict[Tuple[str, str]]): _description_
        targets (Dict[str, Dict[str, np.ndarray]]): _description_

    Returns:
        Dict[str, Dict[str, np.ndarray]]: _description_
    """

    def filtered_genes(tpm_filtered_genes: str) -> List[str]:
        with open(tpm_filtered_genes, newline="") as file:
            return [f"{line[3]}_{tissue}" for line in csv.reader(file, delimiter="\t")]

    for idx, tissue in enumerate(tissues):
        if idx == 0:
            genes = filtered_genes(f"{root_dir}/{tissue}/tpm_filtered_genes.bed")
        else:
            update_genes = filtered_genes(f"{root_dir}/{tissue}/tpm_filtered_genes.bed")
            genes += update_genes

    genes = set(genes)
    for key in targets.keys():
        targets[key] = {
            gene: targets[key][gene] for gene in targets[key].keys() if gene in genes
        }
    return targets


def _convert_coessential_to_gencode(
    coessential: str,
    genesymbol_to_gencode: Dict[str, str],
):
    """_summary_

    Args:
        dir (str): _description_
        coessential (str): _description_
        gencode_ref (str): _description_
    """
    return [
        (genesymbol_to_gencode[line[0]], genesymbol_to_gencode[line[1]], line[2])
        for line in csv.reader(
            open(coessential, newline="", encoding="utf-8-sig"), delimiter="\t"
        )
        if line[0] in genesymbol_to_gencode.keys()
        and line[1] in genesymbol_to_gencode.keys()
    ]


# gencode_ref = "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed"
# gencode_ref = pybedtools.BedTool(gencode_ref)
# genesymbol_to_gencode = genes_from_gencode(gencode_ref)
# co = _convert_coessential_to_gencode(
#     coessential="coessential_pairs.txt",
#     genesymbol_to_gencode=genesymbol_to_gencode,
# )
