#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for genome data processing."""

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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pybedtools
from scipy import stats
import seaborn as sns
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

LOOPFILES = {
    "deepanchor": {
        "aorta": "aorta_deepanchor.bedpe.hg38",
        "hippocampus": "hippocampus_deepanchor.bedpe.hg38",
        "left_ventricle": "left_ventricle_deepanchor.bedpe.hg38",
        "liver": "liver_deepanchor.bedpe.hg38",
        "lung": "lung_deepanchor.bedpe.hg38",
        "mammary": "mammary_deepanchor.bedpe.hg38",
        "pancreas": "pancreas_deepanchor.bedpe.hg38",
        "skeletal_muscle": "skeletal_muscle_deepanchor.bedpe.hg38",
        "skin": "skin_deepanchor.bedpe.hg38",
        "small_intestine": "small_intestine_deepanchor.bedpe.hg38",
    },
    "peakachu": {
        "aorta": "Leung_2015.Aorta.hg38.peakachu-merged.loops",
        "hippocampus": "Schmitt_2016.Hippocampus.hg38.peakachu-merged.loops",
        "left_ventricle": "Leung_2015.VentricleLeft.hg38.peakachu-merged.loops",
        "liver": "Leung_2015.Liver.hg38.peakachu-merged.loops",
        "lung": "Schmitt_2016.Lung.hg38.peakachu-merged.loops",
        "mammary": "Rao_2014.HMEC.hg38.peakachu-merged.loops",
        "pancreas": "Schmitt_2016.Pancreas.hg38.peakachu-merged.loops",
        "skeletal_muscle": "Schmitt_2016.Psoas.hg38.peakachu-merged.loops",
        "skin": "Rao_2014.NHEK.hg38.peakachu-merged.loops",
        "small_intestine": "Schmitt_2016.Bowel_Small.hg38.peakachu-merged.loops",
    },
    "peakachu_deepanchor": {
        "aorta": "aorta_peakachu_deepanchor.hg38.combined_loops",
        "hippocampus": "hippocampus_peakachu_deepanchor.hg38.combined_loops",
        "left_ventricle": "left_ventricle_peakachu_deepanchor.hg38.combined_loops",
        "liver": "liver_peakachu_deepanchor.hg38.combined_loops",
        "lung": "lung_peakachu_deepanchor.hg38.combined_loops",
        "mammary": "mammary_peakachu_deepanchor.hg38.combined_loops",
        "pancreas": "pancreas_peakachu_deepanchor.hg38.combined_loops",
        "skeletal_muscle": "skeletal_muscle_peakachu_deepanchor.hg38.combined_loops",
        "skin": "skin_peakachu_deepanchor.hg38.combined_loops",
        "small_intestine": "small_intestine_peakachu_deepanchor.hg38.combined_loops",
    },
    "deeploop_only": {
        "aorta": "GSE167200_Aorta.top300K_300000_loops.bedpe.hg38",
        "hippocampus": "GSE167200_Hippocampus.top300K_300000_loops.bedpe.hg38",
        "left_ventricle": "GSE167200_LeftVentricle.top300K_300000_loops.bedpe.hg38",
        "liver": "GSE167200_Liver.top300K_300000_loops.bedpe.hg38",
        "lung": "GSE167200_Lung.top300K_300000_loops.bedpe.hg38",
        "pancreas": "GSE167200_Pancreas.top300K_300000_loops.bedpe.hg38",
        "skeletal_muscle": "GSE167200_Psoas_Muscle.top300K_300000_loops.bedpe.hg38",
        "small_intestine": "GSE167200_Small_Intenstine.top300K_300000_loops.bedpe.hg38",
    },
    "deeploop_deepanchor_peakachu": {
        "aorta": "aorta_alloops.bed",
        "hippocampus": "hippocampus_alloops.bed",
        "left_ventricle": "left_ventricle_alloops.bed",
        "liver": "liver_alloops.bed",
        "lung": "lung_alloops.bed",
        "pancreas": "pancreas_alloops.bed",
        "skeletal_muscle": "skeletal_muscle_alloops.bed",
        "small_intestine": "small_intestine_alloops.bed",
    },
}

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

TISSUES_early_testing = [
    "aorta",
    "hippocampus",
    "left_ventricle",
    "liver",
    "lung",
    "pancreas",
    "skeletal_muscle",
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


def time_decorator(
    print_args: bool = False,
    display_arg: str = "",
) -> Callable:
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


class GeneralUtils:
    def __init__(self):
        pass

    @staticmethod
    def parse_yaml(config_file: str) -> Dict[str, Union[str, list]]:
        """Load yaml for parsing"""
        with open(config_file, "r") as stream:
            params = yaml.safe_load(stream)
        return params

    @staticmethod
    def dir_check_make(dir: str) -> None:
        """Utility to make directories only if they do not already exist"""
        try:
            os.makedirs(dir)
        except FileExistsError:
            pass

    @staticmethod
    def check_and_symlink(
        src: str,
        dst: str,
        boolean: bool = False,
    ) -> None:
        try:
            if boolean == True:
                if (bool(src) and os.path.exists(src)) and (not os.path.exists(dst)):
                    os.symlink(src, dst)
            else:
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        except FileExistsError:
            pass

    @staticmethod
    def _listdir_isfile_wrapper(dir: str) -> List[str]:
        """Returns a list of files within the directory"""
        return [file for file in os.listdir(dir) if os.path.isfile(f"{dir}/{file}")]

    @staticmethod
    def _tensor_out_to_array(tensor, idx):
        return np.stack([x[idx].cpu().numpy() for x in tensor], axis=0)


class GenomeDataUtils:
    def __init__(self):
        pass

    @staticmethod
    def chunk_genes(
        genes: List[str],
        chunks: int,
    ) -> Dict[int, List[str]]:
        """Constructs graphs in parallel"""
        ### get list of all gencode V26 genes
        for num in range(0, 5):
            random.shuffle(genes)

        split_list = lambda l, chunks: [
            l[n : n + chunks] for n in range(0, len(l), chunks)
        ]
        split_genes = split_list(genes, chunks)
        return {index: gene_list for index, gene_list in enumerate(split_genes)}

    @staticmethod
    def filtered_genes_from_bed(tpm_filtered_genes: str) -> List[str]:
        with open(tpm_filtered_genes, newline="") as file:
            return [line[3] for line in csv.reader(file, delimiter="\t")]

    @staticmethod
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

    @staticmethod
    def genes_from_gencode(gencode_ref) -> Dict[str, str]:
        """Returns a dict of gencode v26 genes, their ids and associated gene
        symbols
        """
        return {
            line[9].split(";")[3].split('"')[1]: line[3]
            for line in gencode_ref
            if line[0] not in ["chrX", "chrY", "chrM"]
        }

    @staticmethod
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
        df["total"] = df.select_dtypes(np.number).ge(0.1).sum(axis=1)
        df["result"] = df["total"] >= (0.20 * sample_n)
        if return_list == False:
            return [
                f"{gene}_{tissue}" for gene in list(df.loc[df["result"] == True].index)
            ]
        else:
            return list(df.loc[df["result"] == True].index)

    @staticmethod
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
                f"{gene}_{tissue}"
                for gene in list(df.loc[df[df >= tpm].any(axis=1)].index)
            ]
        else:
            return [gene for gene in list(df.loc[df[df >= tpm].any(axis=1)].index)]

    @staticmethod
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
        Filter out genes in a GTEx tissue with less than 0.1 tpm across 20% of
        samples in that tissue. Additionally, we exclude analysis of sex
        chromosomes.

        Returns:
            pybedtools object with +/- <window> windows around that gene
        """
        tpm_filtered_genes = _filter_low_tpm(
            file=tpm_file,
            tissue=tissue,
            return_list=True,
        )
        genes = pybedtools.BedTool(gencode)
        genes_filtered = genes.filter(
            lambda x: x[3] in tpm_filtered_genes
            and x[0] not in ["chrX", "chrY", "chrM"]
        ).saveas()

        if slop:
            return (
                genes_filtered.sort(),
                genes_filtered.slop(g=chromfile, b=window).cut([0, 1, 2, 3]).sort(),
            )
        else:
            return genes_filtered.sort()
            # [x[3] for x in genes_filtered]


class DataVizUtils:
    def __init__(self):
        pass

    @time_decorator(print_args=True)
    def _set_matplotlib_publication_parameters() -> None:
        plt.rcParams.update({"font.size": 7})  # set font size
        plt.rcParams.update({"axes.titlesize": "small"})
        plt.rcParams.update({"font.sans-serif": "Nimbus Sans"})
        # plt.rcParams["font.family"] = "Liberation Sans"  # set font

    @time_decorator(print_args=True)
    def plot_training_losses(
        log: str,
        experiment_name: str,
        model: str,
        layers: int,
        width: int,
        batch_size: int,
        learning_rate: float,
        outdir: str,
    ) -> None:
        plt.figure(figsize=(3.125, 2.25))
        DataVizUtils._set_matplotlib_publication_parameters()

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
            f"Training loss for {experiment_name}, {model}, {layers} layers, lr {learning_rate}, batch size {batch_size}, dimensions {width}",
            wrap=True,
            fontsize=7,
        )
        plt.tight_layout()
        plt.savefig(
            f"{outdir}/{experiment_name}_{model}_{layers}_{width}_{batch_size}_{learning_rate}_dropout_loss.png",
            dpi=300,
        )
        plt.close()

    @time_decorator(print_args=True)
    def plot_predicted_versus_expected(
        expected,
        predicted,
        outdir,
        experiment_name,
        model,
        layers,
        width,
        batch_size,
        learning_rate,
        rmse,
    ):
        plt.figure(figsize=(3, 2.75))
        DataVizUtils._set_matplotlib_publication_parameters()

        sns.regplot(x=expected, y=predicted, scatter_kws={"s": 2, "alpha": 0.1})
        plt.margins(x=0)
        plt.xlabel("Expected Log2 TPM", fontsize=7)
        plt.ylabel("Predicted Log2 TPM", fontsize=7)
        plt.title(
            f"Expected versus predicted for {experiment_name,} {model}, {layers} layers, lr {learning_rate}, batch size {batch_size}, dimensions {width}\nRMSE: {rmse}\nSpearman's R: {stats.spearmanr(expected, predicted)[0]}",
            wrap=True,
            fontsize=7,
        )
        plt.tight_layout()
        plt.savefig(
            f"{outdir}/{experiment_name}_{model}_{layers}_{width}_{batch_size}_{learning_rate}_dropout_performance.png",
            dpi=300,
        )
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
        dir (str): _description_ coessential (str): _description_ gencode_ref
        (str): _description_
    """
    return [
        (genesymbol_to_gencode[line[0]], genesymbol_to_gencode[line[1]], line[2])
        for line in csv.reader(
            open(coessential, newline="", encoding="utf-8-sig"), delimiter="\t"
        )
        if line[0] in genesymbol_to_gencode.keys()
        and line[1] in genesymbol_to_gencode.keys()
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
