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
import logging
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import psutil  # type: ignore
import pybedtools  # type: ignore
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
import torch
import yaml  # type: ignore

from config_handlers import ExperimentConfig
from config_handlers import TissueConfig


# decorator to track execution time
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
                f"Finished {function.__name__} {args_to_print} - \
                Time: {timedelta(seconds=end_time - start_time)}"
            )
            return result

        return _execute

    return _time_decorator_func


# logging setup
def setup_logging() -> logging.Logger:
    """Prepare a logger that prints to stderr."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


# file and directory operations
def dir_check_make(dir: Union[str, Path]) -> None:
    """Utility to make directories only if they do not already exist"""
    with suppress(FileExistsError):
        os.makedirs(dir)


def check_and_symlink(src: Union[str, Path], dst: Path, boolean: bool = False) -> None:
    """Create a symlink from src to dst if it doesn't exist. If boolean is True,
    also check that src exists before creating the symlink.
    """
    if boolean and not os.path.exists(src):
        return
    if not os.path.exists(dst):
        with suppress(FileExistsError):
            os.symlink(src, dst)


def _get_files_in_directory(dir: Path) -> List[str]:
    """Return a list of files within the directory."""
    return [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]


# data loading and saving
def _load_pickle(file_path: str) -> Any:
    """Wrapper to load a pkl"""
    with open(file_path, "rb") as file:
        return pickle.load(file)


def _save_pickle(data: object, file_path: Union[str, Path]) -> None:
    """Wrapper to save a pkl"""
    with open(file_path, "wb") as output:
        pickle.dump(data, output)


def parse_yaml(config_file: str) -> Dict[str, Any]:
    """Load yaml for parsing"""
    with open(config_file, "r") as stream:
        return yaml.safe_load(stream)


# command line operations
def _run_command(command: str, get_output: bool = False) -> Optional[str]:
    """Run a shell command and optionally return its output."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
            shell=True,
        )
        if get_output:
            return result.stdout.strip()
        else:
            print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running command: {command}")
        print(e.output)
        sys.exit(1)
    return None


# slurm operations
def submit_slurm_job(job_script: str, args: str, dependency: Optional[str]) -> str:
    """Submits a SLURM job and returns its job ID."""
    dependency_arg = f"--dependency=afterok:{dependency}" if dependency else ""
    command = f"sbatch {dependency_arg} {job_script} {args}"
    job_id = _run_command(command, get_output=True)
    assert job_id is not None
    return job_id.split()[-1]  # extract job ID


# system operations
def get_physical_cores() -> int:
    """Return physical core count, subtracted by one to account for the main
    process / overhead.
    """
    return psutil.cpu_count(logical=False) - 1


# graph data and gnn operations
def _tensor_out_to_array(tensor, idx):
    return np.stack([x[idx].cpu().numpy() for x in tensor], axis=0)


def _calculate_max_distance_base_graph(bed: List[List[str]]) -> Set[int]:
    """Calculate the max distance between nodes in the base graph. Report the
    max, mean, and median distances for all interaction type data.
    """
    return {
        max(int(line[2]), int(line[6])) - min(int(line[3]), int(line[7]))
        for line in bed
    }


def _combine_and_sort_arrays(edge_index: np.ndarray) -> np.ndarray:
    """Combines stored edge index and returns dedupe'd array of nodes"""
    combined = np.concatenate((edge_index[0], edge_index[1]))
    return np.unique(combined)


# genome data utils
class ScalerUtils:
    """Utility class for scaling node features, as the modules for scaling share
    most of the same args"""

    def __init__(self) -> None:
        """Instantiate the class."""
        args = self._parse_args()
        self.experiment_config = ExperimentConfig.from_yaml(args.experiment_config)
        self.split_name = args.split_name
        self.onehot_node_feats = self.experiment_config.onehot_node_feats
        self.file_prefix = f"{self.experiment_config.experiment_name}_{self.experiment_config.graph_type}"

        # directories
        self.graph_dir = self.experiment_config.graph_dir
        self.split_dir = self.graph_dir / self.split_name
        self.scaler_dir = self.split_dir / "scalers"

        # files
        self.idxs = self.graph_dir / f"{self.file_prefix}_graph_idxs.pkl"
        self.graph = self.graph_dir / f"{self.file_prefix}_graph.pkl"
        self.split = self.split_dir / "training_split_combined.pkl"

    def load_from_file(self, file_path: Union[Path, str]) -> Dict[str, Any]:
        """Load data from a file."""
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def load_idxs(self) -> Dict[str, int]:
        """Load indexes from file."""
        return self.load_from_file(self.idxs)

    def load_graph(self) -> Dict[str, Any]:
        """Load graph from file."""
        return self.load_from_file(self.graph)

    def load_split(self) -> Dict[str, Any]:
        """Load split from file."""
        return self.load_from_file(self.split)

    @staticmethod
    def _parse_args():
        """Get arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--experiment_config",
            type=str,
            help="Path to .yaml file with experimental conditions",
        )
        parser.add_argument("--split_name", type=str)
        return parser.parse_args()


def _get_chromatin_loop_file(
    experiment_config: ExperimentConfig, tissue_config: TissueConfig
) -> str:
    """Returns the specified loop file"""
    method, resolution = experiment_config.baseloops.split("_")
    return f"{experiment_config.baseloop_dir}/{method}/{resolution}/{tissue_config.resources['tissue']}_loops.bedpe"


def _dataset_split_name(
    test_chrs: Optional[List[str]] = None,
    val_chrs: Optional[List[str]] = None,
    tpm_filter: Union[float, int] = 0.1,
    percent_of_samples_filter: float = 0.2,
) -> str:
    """Constructs a name for the dataset based on the chromosome split and gene
    filter arguments.
    """
    chrs = []

    if test_chrs and val_chrs:
        chrs.append(
            f"test_{('-').join(map(str, test_chrs))}_val_{('-').join(map(str, val_chrs))}"
        )
    elif test_chrs:
        chrs.append(f"test_{('-').join(map(str, test_chrs))}")
    elif val_chrs:
        chrs.append(f"val_{('-').join(map(str, val_chrs))}")
    else:
        chrs.append("random_assign")

    return f"tpm_{tpm_filter}_samples_{percent_of_samples_filter}_{''.join(chrs).replace('chr', '')}"


def chunk_genes(
    genes: List[str],
    chunks: int,
) -> Dict[int, List[str]]:
    """Constructs graphs in parallel"""
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

    Examples:
    >>> prepare_tss_file(
            tss_file="refTSS_v4.1_human_coordinate.hg38.bed.txt",
            annotation_file="refTSS_v4.1_human_hg38_annotation.txt",
            gencode_ref="gencode_v26_genes_only_with_GTEx_targets.bed",
            savedir="/reftss",
        )
    """
    _reftss_cut_cols(annotation_file)
    genesymbol_to_gencode = genes_from_gencode(pybedtools.BedTool(gencode_ref))
    tss = pybedtools.BedTool(tss_file)
    maps = _tss_tuples_to_dict(
        _tss_to_gene_tuples(f"{annotation_file}.cut"),
        genesymbol_to_gencode=genesymbol_to_gencode,
    )

    bed: List[List[str]] = []
    for line in tss:
        if line[3] in maps:
            bed.extend(
                [line[0], line[1], line[2], f"tss_{line[3]}_{value}"]
                for value in maps[line[3]]
            )
        else:
            bed.append([line[0], line[1], line[2], f"tss_{line[3]}"])

    bed = pybedtools.BedTool(bed).saveas(f"{savedir}/tss_parsed_hg38.bed")


def _map_genesymbol_to_tss(
    tss_path: str, annotation_path: str
) -> List[tuple[str, str, str, str]]:
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


# plotting utilities
@time_decorator(print_args=True)
def _set_matplotlib_publication_parameters() -> None:
    plt.rcParams.update(
        {"font.size": 7, "axes.titlesize": "small", "font.sans-serif": "Nimbus Sans"}
    )


@time_decorator(print_args=True)
def plot_training_losses(
    outfile: str,
    savestr: str,
    log: str,
) -> None:
    """Plots training losses from training log"""
    plt.figure(figsize=(3.125, 2.25))
    _set_matplotlib_publication_parameters()

    losses: Dict[str, List[float]] = {"Train": [], "Test": [], "Validation": []}
    with open(log, newline="") as file:
        reader = csv.reader(file, delimiter=":")
        for line in reader:
            for substr in line:
                for key in losses:
                    if key in substr:
                        losses[key].append(float(line[-1].split(" ")[-1]))

    # # remove last item in train
    # try:
    #     loss_df = pd.DataFrame(losses)
    # except ValueError:
    #     losses["Train"] = losses["Train"][:-1]
    if len(losses["Train"]) > len(losses["Test"]):
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
        f"Expected versus predicted for {savestr}\n"
        f"RMSE: {rmse}\n"
        f"Spearman's R: {stats.spearmanr(expected, predicted)[0]}\n"
        f"Pearson: {stats.pearsonr(expected, predicted)[0]}",
        wrap=True,
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
