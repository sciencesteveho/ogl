# sourcery skip: do-not-use-staticmethod
#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for omics graph learning modules."""

import argparse
from collections import defaultdict
from contextlib import suppress
import csv
from datetime import timedelta
import functools
import inspect
import os
import pathlib
import pickle
import random
import subprocess
import time
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pybedtools
from scipy import stats
import seaborn as sns
import torch
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

NODE_FEAT_IDXS = {
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
}

POSSIBLE_NODES = [
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

TISSUES = [
    "aorta",
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


def _generate_deeploop_dict(resolution: Union[int, str]) -> Dict[str, str]:
    """Generate a dictionary of deeploop filenames for a given resolution"""
    special_tissues = {
        "left_ventricle": "leftventricle",
        "skeletal_muscle": "psoas_muscle",
    }
    tissues = [
        "aorta",
        "hippocampus",
        "left_ventricle",
        "liver",
        "lung",
        "pancreas",
        "skeletal_muscle",
        "small_intestine",
    ]
    return {
        tissue: f"{special_tissues.get(tissue, tissue)}_{resolution}_pixels.hg38"
        for tissue in tissues
    }


class ScalerUtils:
    """Utility class for scaling node features, as the modules for scaling share
    most of the smae args"""

    @staticmethod
    def _parse_args():
        """Get arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--feat", type=int, default=0)
        parser.add_argument("-g", "--graph_type", type=str)
        parser.add_argument(
            "--experiment_config",
            type=str,
            help="Path to .yaml file with experimental conditions",
        )
        parser.add_argument("--split_name", type=str)
        return parser.parse_args()

    @staticmethod
    def _unpack_params(params: Dict[str, Union[str, List[str], Dict[str, str]]]):
        """Unpack params from the .yaml"""
        experiment_name = params["experiment_name"]
        working_directory = params["working_directory"]
        # target_params = params["training_targets"]
        return (
            experiment_name,
            pathlib.Path(working_directory),
        )

    @staticmethod
    def _handle_scaler_prep():
        """Handle scaler prep"""
        args = ScalerUtils._parse_args()
        params = parse_yaml(args.experiment_config)
        experiment_name, working_directory = ScalerUtils._unpack_params(params)

        working_path = pathlib.Path(working_directory)
        graph_dir = working_path / experiment_name / "graphs"
        prefix = f"{experiment_name}_{args.graph_type}"
        return (
            args.feat,
            graph_dir / args.split_name,
            graph_dir / args.split_name / "scalers",
            prefix,
            graph_dir / prefix,
            f"{prefix}_{args.split_name}_graph",
        )

    @staticmethod
    def _load_graph_data(
        graphdir_prefix: str,
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Load graph data from files."""
        idxs_file_path = f"{graphdir_prefix}_graph_idxs.pkl"
        graph_file_path = f"{graphdir_prefix}_graph.pkl"
        with open(idxs_file_path, "rb") as idxs_file, open(
            graph_file_path, "rb"
        ) as graph_file:
            idxs = pickle.load(idxs_file)
            graph = pickle.load(graph_file)
        return idxs, graph


def _load_pickle(file_path: str) -> Any:
    """Wrapper to load a pkl"""
    with open(file_path, "rb") as file:
        return pickle.load(file)


def _save_pickle(data: object, file_path: str) -> Any:
    """Wrapper to save a pkl"""
    with open(file_path, "wb") as output:
        pickle.dump(data, output)


def time_decorator(print_args: bool = False, display_arg: str = "") -> Callable:
    """Decorator to time functions.

    Args:
        print_args (bool, optional): Whether to print the function arguments.
        Defaults to False. display_arg (str, optional): The argument to display
        in the print statement. Defaults to "".

    Returns:
        Callable: The decorated function.
    """

    def _time_decorator_func(function: Callable) -> Callable:
        @functools.wraps(function)
        def _execute(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            fxn_args = inspect.signature(function).bind(*args, **kwargs).arguments
            result = function(*args, **kwargs)
            end_time = time.monotonic()
            args_to_print = list(fxn_args.values()) if print_args else display_arg
            print(
                f"Finished {function.__name__} {args_to_print} - Time: {timedelta(seconds=end_time - start_time)}"
            )
            return result

        return _execute

    return _time_decorator_func


def parse_yaml(config_file: str) -> Dict[str, Union[str, list]]:
    """Load yaml for parsing"""
    with open(config_file, "r") as stream:
        params = yaml.safe_load(stream)
    return params


def dir_check_make(dir: str) -> None:
    """Utility to make directories only if they do not already exist"""
    with suppress(FileExistsError):
        os.makedirs(dir)


def _dataset_split_name(
    test_chrs: List[int] = None,
    val_chrs: List[int] = None,
    tpm_filter: Union[float, int] = 0.1,
    percent_of_samples_filter: float = 0.2,
) -> None:
    """Save the partitioning split to a file based on provided chromosome
    information.

    Args:
        save_dir (str): The directory where the split file will be saved.
        test_chrs (List[int]): A list of test chromosomes.
        val_chrs (List[int]): A list of validation chromosomes.
        split (Dict[str, List[str]]): A dictionary containing the split data.

    Returns:
        None
    """
    chrs = []

    if test_chrs and val_chrs:
        chrs.append(f"test_{('-').join(test_chrs)}_val_{('-').join(val_chrs)}")
    elif test_chrs:
        chrs.append(f"test_{('-').join(test_chrs)}")
    elif val_chrs:
        chrs.append(f"val_{('-').join(val_chrs)}")
    else:
        chrs.append("random_assign")

    return f"tpm_{tpm_filter}_samples_{percent_of_samples_filter}_{''.join(chrs).replace('chr', '')}"


def check_and_symlink(
    src: str,
    dst: str,
    boolean: bool = False,
) -> None:
    """Check if a symlink exists at the destination path and create a symlink
    from the source path to the destination path if it doesn't exist.

    Args:
        src (str): The source path of the symlink. dst (str): The destination
        path of the symlink. boolean (bool, optional): A boolean flag. If True,
        the symlink is created only if the source path exists and the
        destination path doesn't exist. If False, the symlink is created if the
        destination path doesn't exist. Defaults to False.
    """
    with suppress(FileExistsError):
        if boolean:
            if (bool(src) and os.path.exists(src)) and (not os.path.exists(dst)):
                os.symlink(src, dst)
        elif not os.path.exists(dst):
            os.symlink(src, dst)


def _listdir_isfile_wrapper(dir: str) -> List[str]:
    """Returns a list of files within the directory"""
    return [file for file in os.listdir(dir) if os.path.isfile(f"{dir}/{file}")]


def _tensor_out_to_array(tensor, idx):
    return np.stack([x[idx].cpu().numpy() for x in tensor], axis=0)


def chunk_genes(
    genes: List[str],
    chunks: int,
) -> Dict[int, List[str]]:
    """Constructs graphs in parallel"""
    ### get list of all gencode V26 genes
    for _ in range(5):
        random.shuffle(genes)

    split = lambda l, chunks: [l[n : n + chunks] for n in range(0, len(l), chunks)]
    split_genes = split(genes, chunks)
    return dict(enumerate(split_genes))


def filtered_genes_from_bed(tpm_filtered_genes: str) -> List[str]:
    """Extracts the gene names from a filtered genes BED file."""
    with open(tpm_filtered_genes, newline="") as file:
        return [line[3] for line in csv.reader(file, delimiter="\t")]


def genes_from_gencode(gencode_ref: pybedtools.BedTool) -> Dict[str, str]:
    """Returns a dict of gencode v26 genes, their ids and associated gene
    symbols
    """
    return {
        line[9].split(";")[3].split('"')[1]: line[3]
        for line in gencode_ref
        if line[0] not in ["chrX", "chrY", "chrM"]
    }


def _reftss_cut_cols(file: str) -> None:
    """Cuts the first and eighth columns of the refTSS annotation file to
    produce a file with only the TSS and gene symbol"""
    cmd = f"cut -f1,8 {file} > {file}.cut"
    subprocess.run(cmd, stdout=None, shell=True)


def _tss_to_gene_tuples(file: str) -> List[Tuple[str, str]]:
    """Read refTSS annotation file and return a list of tuples matching TSS to
    each gene symbol."""
    with open(file) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        return [(row[0], col) for row in reader for col in row[1].split(" ") if col]


def _tss_tuples_to_dict(
    tss_tuples: List[Tuple[str, str]],
    genesymbol_to_gencode: Dict[str, str],
) -> Dict[str, List[str]]:
    """Map tuples to a dictionary with the first element as the key"""
    key_mappings = defaultdict(list)
    for key, value in tss_tuples:
        if value in genesymbol_to_gencode:
            key_mappings[key].append(genesymbol_to_gencode[value])
    return key_mappings


def prepare_tss_file(
    tss_file: str, annotation_file: str, gencode_ref: str, savedir: str
) -> None:
    """Prepares parsed TSS file, where each TSS is linked with its gencode gene.
    If multiple genes are associated with the TSS, the tss is split across
    multiple lines.

    Args:
        tss_file (str): The path to the TSS file.
        annotation_file (str): The path to the annotation file.
        gencode_ref (str): The path to the GENCODE reference file.
        savedir (str): The directory to save the output file.

    Returns:
        None

    Example:
        prepare_tss_file(
            tss_file="refTSS_v4.1_human_coordinate.hg38.bed.txt",
            annotation_file="refTSS_v4.1_human_hg38_annotation.txt",
            gencode_ref="/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed",
            savedir="/ocean/projects/bio210019p/stevesho/data/bedfile_preparse/reftss",
        )
    """
    _reftss_cut_cols(annotation_file)
    genesymbol_to_gencode = genes_from_gencode(pybedtools.BedTool(gencode_ref))
    tss = pybedtools.BedTool(tss_file)
    maps = _tss_tuples_to_dict(
        _tss_to_gene_tuples(f"{annotation_file}.cut"),
        genesymbol_to_gencode=genesymbol_to_gencode,
    )

    bed = []
    for line in tss:
        if line[3] in maps:
            bed.extend(
                [line[0], line[1], line[2], f"tss_{line[3]}_{value}"]
                for value in maps[line[3]]
            )
        else:
            bed.append([line[0], line[1], line[2], f"tss_{line[3]}"])

    bed = pybedtools.BedTool(bed).saveas(f"{savedir}/tss_parsed_hg38.bed")


def genes_from_gff(gff: str) -> List[str]:
    """Get list of gtex genes from GFF file

    Args:
        gff (str): /path/to/genes.gtf

    Returns:
        List[str]: genes
    """
    with open(gff, newline="") as file:
        return {
            line[3]: line[0]
            for line in csv.reader(file, delimiter="\t")
            if line[0] not in ["chrX", "chrY", "chrM"]
        }


@time_decorator(print_args=True)
def _filter_low_tpm(
    file: str,
    tissue: str,
    return_list: bool = False,
) -> List[str]:
    """
    Filter genes according to the following criteria: (A) Only keep genes
    expressing >= 0.1 TPM across 20% of samples in that tissue
    """
    df = pd.read_table(file, index_col=0, header=[2])
    sample_n = len(df.columns)
    df["total"] = df.select_dtypes(np.number).ge(1).sum(axis=1)
    df["result"] = df["total"] >= (0.20 * sample_n)
    if not return_list:
        return [f"{gene}_{tissue}" for gene in list(df.loc[df["result"] == True].index)]
    else:
        return list(df.loc[df["result"] == True].index)


@time_decorator(print_args=True)
def filter_genes_by_tpm(
    gencode: str,
    tpm_file: str,
    tpm_filter: Union[float, int],
    percent_of_samples_filter: float,
) -> Tuple[pybedtools.BedTool, List[str]]:
    """
    Filter out genes in a GTEx tissue with less than 1 tpm across 20% of
    samples in that tissue. Additionally, we exclude analysis of sex
    chromosomes.

    Returns:
        pybedtools object with +/- <window> windows around that gene
    """
    df = pd.read_table(tpm_file, index_col=0, header=[2])
    sample_n = len(df.columns)
    df["total"] = df.select_dtypes(np.number).ge(tpm_filter).sum(axis=1)
    df["result"] = df["total"] >= (percent_of_samples_filter * sample_n)
    tpm_filtered_genes = list(df.loc[df["result"] == True].index)

    genes = pybedtools.BedTool(gencode)
    genes_filtered = genes.filter(
        lambda x: x[3] in tpm_filtered_genes and x[0] not in ["chrX", "chrY", "chrM"]
    ).saveas()

    return genes_filtered.sort()


@time_decorator(print_args=True)
def _set_matplotlib_publication_parameters() -> None:
    plt.rcParams.update({"font.size": 7})  # set font size
    plt.rcParams.update({"axes.titlesize": "small"})
    plt.rcParams.update({"font.sans-serif": "Nimbus Sans"})
    # plt.rcParams["font.family"] = "Liberation Sans"  # set font


@time_decorator(print_args=True)
def plot_training_losses(
    outfile: str,
    savestr: str,
    log: str,
) -> None:
    """Plots training losses from training log"""
    plt.figure(figsize=(3.125, 2.25))
    _set_matplotlib_publication_parameters()

    losses = {"Train": [], "Test": [], "Validation": []}
    with open(log, newline="") as file:
        reader = csv.reader(file, delimiter=":")
        for line in reader:
            for substr in line:
                for key in losses:
                    if key in substr:
                        losses[key].append(float(line[-1].split(" ")[-1]))

    # remove last item in train
    try:
        losses = pd.DataFrame(losses)
    except ValueError:
        losses["Train"] = losses["Train"][:-1]

    sns.lineplot(data=losses)
    plt.margins(x=0)
    plt.xlabel("Epoch", fontsize=7)
    plt.ylabel("MSE Loss", fontsize=7)
    plt.title(
        f"Training loss for {savestr}",
        wrap=True,
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


@time_decorator(print_args=True)
def plot_predicted_versus_expected(
    outfile: str,
    savestr: str,
    predicted: torch.Tensor,
    expected: torch.Tensor,
    rmse: torch.Tensor,
) -> None:
    """Plots predicted versus expected values for a given model"""
    plt.figure(figsize=(3.15, 2.95))
    _set_matplotlib_publication_parameters()

    sns.regplot(x=expected, y=predicted, scatter_kws={"s": 2, "alpha": 0.1})
    plt.margins(x=0)
    plt.xlabel("Expected Log2 TPM", fontsize=7)
    plt.ylabel("Predicted Log2 TPM", fontsize=7)
    plt.title(
        f"Expected versus predicted for {savestr}\
            \nRMSE: {rmse}\
            \nSpearman's R: {stats.spearmanr(expected, predicted)[0]}\
            \nPearson: {stats.pearsonr(expected, predicted)[0]}",
        wrap=True,
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


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
    graphs = [
        nx.read_gml(f"{graph_dir}/{tissue}/{tissue}_{graph_type}_graph.gml")
        for tissue in tissue_list
    ]
    return nx.compose_all(graphs)


def _combined_graph_arrays(
    tissue_list: List[str],
    graph_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine graph arrays from multiple tissues"""
    feats = []
    for _ in tissue_list:
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
    genesymbols = {
        line[0]: line[7] for line in csv.reader(open(annotation_path), delimiter="\t")
    }

    return [
        (line[0], line[1], line[2], f"tss_{line[3]}_{genesymbols[line[3]]}")
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


def _convert_coessential_to_gencode(
    coessential: str,
    genesymbol_to_gencode: Dict[str, str],
):
    """_summary_

    Args:
        dir (str): _description_ coessential (str): _description_ gencode_ref
        (str): _description_
    """
    return [
        (
            genesymbol_to_gencode[line[0]],
            genesymbol_to_gencode[line[1]],
            line[2],
        )
        for line in csv.reader(
            open(coessential, newline="", encoding="utf-8-sig"), delimiter="\t"
        )
        if line[0] in genesymbol_to_gencode and line[1] in genesymbol_to_gencode
    ]


def gene_list_from_graphs(root_dir: str, tissue: str) -> List[str]:
    """Returns a list of genes with constructed graphs, avoiding genes that may
    not have edges in smaller window"""
    directory = f"{root_dir}/{tissue}/parsing/graphs"
    return [gene.split("_")[0] for gene in os.listdir(directory)]


def _string_list(arg):
    """Helper function to pass comma separated list of strings from argparse as
    list
    """
    return arg.split(",")


def _combine_and_sort_arrays(edge_index: np.ndarray) -> np.ndarray:
    """Combines stored edge index and returns dedupe'd array of nodes"""
    combined = np.concatenate((edge_index[0], edge_index[1]))
    return np.unique(combined)


def _open_graph(g_path: str, indexes: str, split: str, targets: str):
    """Open pickled graph, indexes, split, and targets"""
    with open(g_path, "rb") as f:
        graph = pickle.load(f)
    with open(indexes, "rb") as f:
        indexes = pickle.load(f)
    with open(split, "rb") as f:
        split = pickle.load(f)
    with open(targets, "rb") as f:
        targets = pickle.load(f)
    return graph, indexes, split, targets


def _get_indexes(
    split: List[str], indexes: Dict[str, int]
) -> Tuple[List[int], List[str]]:
    present = [indexes[i] for i in split if i in indexes]
    not_present = [i for i in split if i not in indexes]
    return present, not_present


def _get_split_indexes(
    split: Dict[str, List[str]], indexes: Dict[str, int]
) -> Tuple[List[int], List[int], List[int], List[str], List[str], List[str]]:
    """Lorem Ipsum"""
    results = []
    for key in ("train", "validation", "test"):
        results.extend(_get_indexes(split[key], indexes))
    return tuple(results)


def _get_dict_subset_by_split(idxs, count_dict):
    """Lorem Ipsum"""
    return {idx: count_dict[idx] for idx in idxs}


def _average_edges_per_expression_node(edge_count_dict):
    """Sum edge counts (the values in the dictionary) and divide by the number
    of keys (nodes)"""
    return sum(edge_count_dict.values()) / len(edge_count_dict.keys())


def _get_targets_for_train_list(genes, targets):
    """Lorem Ipsum"""
    return {gene: targets[gene][0] for gene in genes}


# def filter_target_split(
#     root_dir: str,
#     tissues: Dict[Tuple[str, str], None],
#     targets: Dict[str, Dict[str, np.ndarray]],
# ) -> Dict[str, Dict[str, np.ndarray]]:
#     """Filters and only keeps targets that pass the TPM filter of >1 TPM across
#     20% of samples

#     Args:
#         root_dir (str): The root directory.
#         tissues (Dict[Tuple[str, str], None]): The tissues.
#         targets (Dict[str, Dict[str, np.ndarray]]): The targets.

#     Returns:
#         Dict[str, Dict[str, np.ndarray]]: The filtered targets.
#     """

#     def filtered_genes(tpm_filtered_genes: str) -> List[str]:
#         with open(tpm_filtered_genes, newline="") as file:
#             return [f"{line[3]}_{tissue}" for line in csv.reader(file, delimiter="\t")]

#     for idx, tissue in enumerate(tissues):
#         if idx == 0:
#             genes = filtered_genes(f"{root_dir}/{tissue}/tpm_filtered_genes.bed")
#         else:
#             update_genes = filtered_genes(f"{root_dir}/{tissue}/tpm_filtered_genes.bed")
#             genes += update_genes

#     genes = set(genes)
#     for key in targets:
#         targets[key] = {
#             gene: targets[key][gene] for gene in targets[key].keys() if gene in genes
#         }
#     return targets


# TISSUES_early_testing = [
#     "aorta",
#     "hippocampus",
#     "left_ventricle",
#     "liver",
#     "lung",
#     "pancreas",
#     "skeletal_muscle",
#     "small_intestine",
#     # "hela",
#     # "k562",
#     # "npc",
# ]

# ONEHOT_EDGETYPE = {
#     "g_e": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "gene-enhancer"
#     "g_p": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "gene-promoter"
#     "g_d": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "gene-dyadic"
#     "g_se": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "gene-superenhancer"
#     "p_e": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # "promoter-enhancer"
#     "p_d": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # "promoter-dyadic"
#     "p_se": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # "promoter-superenhancer"
#     "g_g": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # "gene-gene"
#     "ppi": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # "protein-protein"
#     "mirna": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # "mirna-gene"
#     "tf_marker": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # "tf-marker"
#     "circuits": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # "circuits"
#     "local": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # "local"
# }


# LOOPFILES = {
#     "deepanchor": {
#         "aorta": "aorta_deepanchor.bedpe.hg38",
#         "hippocampus": "hippocampus_deepanchor.bedpe.hg38",
#         "left_ventricle": "left_ventricle_deepanchor.bedpe.hg38",
#         "liver": "liver_deepanchor.bedpe.hg38",
#         "lung": "lung_deepanchor.bedpe.hg38",
#         "mammary": "mammary_deepanchor.bedpe.hg38",
#         "pancreas": "pancreas_deepanchor.bedpe.hg38",
#         "skeletal_muscle": "skeletal_muscle_deepanchor.bedpe.hg38",
#         "skin": "skin_deepanchor.bedpe.hg38",
#         "small_intestine": "small_intestine_deepanchor.bedpe.hg38",
#     },
#     "peakachu": {
#         "aorta": "Leung_2015.Aorta.hg38.peakachu-merged.loops",
#         "hippocampus": "Schmitt_2016.Hippocampus.hg38.peakachu-merged.loops",
#         "left_ventricle": "Leung_2015.VentricleLeft.hg38.peakachu-merged.loops",
#         "liver": "Leung_2015.Liver.hg38.peakachu-merged.loops",
#         "lung": "Schmitt_2016.Lung.hg38.peakachu-merged.loops",
#         "mammary": "Rao_2014.HMEC.hg38.peakachu-merged.loops",
#         "pancreas": "Schmitt_2016.Pancreas.hg38.peakachu-merged.loops",
#         "skeletal_muscle": "Schmitt_2016.Psoas.hg38.peakachu-merged.loops",
#         "skin": "Rao_2014.NHEK.hg38.peakachu-merged.loops",
#         "small_intestine": "Schmitt_2016.Bowel_Small.hg38.peakachu-merged.loops",
#     },
#     "peakachu_deepanchor": {
#         "aorta": "aorta_peakachu_deepanchor.hg38.combined_loops",
#         "hippocampus": "hippocampus_peakachu_deepanchor.hg38.combined_loops",
#         "left_ventricle": "left_ventricle_peakachu_deepanchor.hg38.combined_loops",
#         "liver": "liver_peakachu_deepanchor.hg38.combined_loops",
#         "lung": "lung_peakachu_deepanchor.hg38.combined_loops",
#         "mammary": "mammary_peakachu_deepanchor.hg38.combined_loops",
#         "pancreas": "pancreas_peakachu_deepanchor.hg38.combined_loops",
#         "skeletal_muscle": "skeletal_muscle_peakachu_deepanchor.hg38.combined_loops",
#         "skin": "skin_peakachu_deepanchor.hg38.combined_loops",
#         "small_intestine": "small_intestine_peakachu_deepanchor.hg38.combined_loops",
#     },
#     "deeploop_only": {
#         "aorta": "GSE167200_Aorta.top300K_300000_loops.bedpe.hg38",
#         "hippocampus": "GSE167200_Hippocampus.top300K_300000_loops.bedpe.hg38",
#         "left_ventricle": "GSE167200_LeftVentricle.top300K_300000_loops.bedpe.hg38",
#         "liver": "GSE167200_Liver.top300K_300000_loops.bedpe.hg38",
#         "lung": "GSE167200_Lung.top300K_300000_loops.bedpe.hg38",
#         "pancreas": "GSE167200_Pancreas.top300K_300000_loops.bedpe.hg38",
#         "skeletal_muscle": "GSE167200_Psoas_Muscle.top300K_300000_loops.bedpe.hg38",
#         "small_intestine": "GSE167200_Small_Intenstine.top300K_300000_loops.bedpe.hg38",
#     },
#     "deeploop_deepanchor_peakachu": {
#         "aorta": "aorta_alloops.bed",
#         "hippocampus": "hippocampus_alloops.bed",
#         "left_ventricle": "left_ventricle_alloops.bed",
#         "liver": "liver_alloops.bed",
#         "lung": "lung_alloops.bed",
#         "pancreas": "pancreas_alloops.bed",
#         "skeletal_muscle": "skeletal_muscle_alloops.bed",
#         "small_intestine": "small_intestine_alloops.bed",
#     },
# }
