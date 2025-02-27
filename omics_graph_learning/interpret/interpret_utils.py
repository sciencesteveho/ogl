#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Common utility functions for interpretability and perturbation
experiments.
"""


import argparse
import os
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.graph_to_pytorch import GraphToPytorch
from omics_graph_learning.interpret.perturb_runner import PerturbRunner
from omics_graph_learning.utils.common import _dataset_split_name
from omics_graph_learning.utils.config_handlers import ExperimentConfig
from omics_graph_learning.utils.constants import RANDOM_SEEDS


def calculate_log2_fold_change(
    baseline_prediction: float, perturbation_prediction: float
) -> float:
    """Calculate the log2 fold change from log2-transformed values."""
    log2_fold_change = perturbation_prediction - baseline_prediction
    return 2**log2_fold_change - 1


def compute_mean_abs_diff(sub: pd.DataFrame) -> float:
    """Compute the mean absolute difference between prediction and label."""
    return np.mean(np.abs(sub["prediction"] - sub["label"]))


def classify_tpm(x: float) -> str:
    """Bin gene expression values into high, medium, and low categories."""
    if x >= 5:
        return "high"
    elif x >= 1:
        return "medium"
    elif x > 0:
        return "low"
    return "none"  # tpm <= 0


def map_symbol(gene_id: str, gencode_to_symbol: Dict[str, str]) -> str:
    """Map gene IDs to gene symbols."""
    ensg_id = gene_id.split("_")[0]
    return gencode_to_symbol.get(ensg_id, gene_id)


def load_gencode_lookup(filepath: str) -> Dict[str, str]:
    """Load the Gencode-to-gene-symbol lookup table."""
    gencode_to_symbol = {}
    with open(filepath, "r") as f:
        for line in f:
            gencode, symbol = line.strip().split("\t")
            gencode_to_symbol[symbol] = gencode
    return gencode_to_symbol


def combine_masks(data: Data) -> Data:
    """Combine test/train/val masks into one."""
    data.all_mask = data.test_mask | data.train_mask | data.val_mask
    return data


def get_gene_idx_mapping(idxs: Dict[str, int]) -> Tuple[Dict[int, str], List[int]]:
    """Map 'ENSG...' nodes to a dict of node_idx->gene_id and a list of gene_indices."""
    gene_idxs = {k: v for k, v in idxs.items() if "ENSG" in k}
    node_idx_to_gene_id = {v: k for k, v in gene_idxs.items()}
    gene_indices = list(gene_idxs.values())
    return node_idx_to_gene_id, gene_indices


def load_data_and_model(
    lookup_file: str,
    graph_file: str,
    idx_file: str,
    model_file: str,
    device: torch.device,
) -> Tuple[
    Data,
    PerturbRunner,
    Dict[int, str],
    List[int],
    Dict[int, str],
    Dict[str, int],
    Dict[str, str],
]:
    """Load model, Data object, and index mappings."""
    # load gencode: symbol table
    symbol_to_gencode = load_gencode_lookup(lookup_file)

    # load PyG data
    data = torch.load(graph_file).to(device)

    # load node index dictionary
    with open(idx_file, "rb") as f:
        idxs = pickle.load(f)

    node_idx_to_gene_id, gene_indices = get_gene_idx_mapping(idxs)

    idxs_inv = {v: k for k, v in idxs.items()}  # inverse mapping

    # load model via PerturbRunner
    model = PerturbRunner.load_model(
        checkpoint_file=model_file,
        map_location=device,
        model="GAT",
        activation="gelu",
        in_size=42,
        embedding_size=200,
        gnn_layers=2,
        shared_mlp_layers=2,
        heads=2,
        dropout_rate=0.1,
        residual="distinct_source",
        attention_task_head=False,
    )
    runner = PerturbRunner(model=model, device=device, data=data)

    return (
        data,
        runner,
        node_idx_to_gene_id,
        gene_indices,
        idxs_inv,
        idxs,
        symbol_to_gencode,
    )


def _interpret_setup(args: argparse.Namespace) -> Tuple[
    Data,
    torch.device,
    PerturbRunner,
    Dict[int, str],
    List[int],
    Dict[int, str],
    Dict[str, int],
    Dict[str, str],
    Path,
]:
    """Prepare shared setup for interpretation experiments: parses args, sets
    device, random seed, derives paths, loads graph data in pyg, runner, and
    model.
    """
    # set seed and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = RANDOM_SEEDS[args.run_number - 1]
    torch.manual_seed(seed)

    # load experiment config to derive paths
    experiment_config = ExperimentConfig.from_yaml(args.experiment_config)
    root_dir, split_name, experiment_name, idx_file, gene_id_lookup = _derive_paths(
        experiment_config=experiment_config
    )

    # make outpath
    outpath = root_dir / "interpretation" / experiment_name
    os.makedirs(outpath, exist_ok=True)

    # get graph file
    graph_file = _create_pyg_data(
        experiment_config=experiment_config,
        outpath=outpath,
        split_name=split_name,
        experiment_name=experiment_name,
    )

    # get model checkpoint
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = experiment_config.experiment_name

    model_checkpoint = (
        f"{root_dir}/models/{model_name}"
        f"/run_{args.run_number}/{args.model_checkpoint}"
    )

    # load data
    (
        data,
        runner,
        node_idx_to_gene_id,
        gene_indices,
        idxs_inv,
        idxs,
        symbol_to_gencode,
    ) = load_data_and_model(
        lookup_file=gene_id_lookup,
        graph_file=graph_file,
        idx_file=idx_file,
        model_file=model_checkpoint,
        device=device,
    )

    return (
        data,
        device,
        runner,
        node_idx_to_gene_id,
        gene_indices,
        idxs_inv,
        idxs,
        symbol_to_gencode,
        outpath,
    )


def get_baseline_predictions(
    data: Data,
    runner: PerturbRunner,
    mask: str = "all",
) -> pd.DataFrame:
    """Evaluate the model on a given mask and return predictions as a
    DataFrame.
    """
    data = combine_masks(data)
    test_loader = NeighborLoader(
        data,
        num_neighbors=[data.avg_edges * 2] * 2,
        batch_size=64,
        input_nodes=getattr(data, f"{mask}_mask"),
        shuffle=False,
    )

    (
        regression_outs,
        regression_labels,
        node_indices,
        classification_outs,
        classification_labels,
    ) = runner.evaluate(data_loader=test_loader, epoch=0, mask=mask)

    # ensure shape alignment
    regression_outs = regression_outs.squeeze()
    regression_labels = regression_labels.squeeze()
    node_indices = node_indices.squeeze()
    classification_outs = classification_outs.squeeze()
    classification_labels = classification_labels.squeeze()

    assert (
        regression_outs.shape[0]
        == regression_labels.shape[0]
        == node_indices.shape[0]
        == classification_outs.shape[0]
        == classification_labels.shape[0]
    ), "Mismatch in tensor shapes."

    return pd.DataFrame(
        {
            "node_idx": node_indices.cpu().numpy(),
            "prediction": regression_outs.cpu().numpy(),
            "label": regression_labels.cpu().numpy(),
            "class_logits": classification_outs.cpu().numpy(),
            "class_label": classification_labels.cpu().numpy(),
        }
    )


def get_baseline_predictions_k_hop(
    data: Data,
    runner: PerturbRunner,
    k: int = 3,
) -> pd.DataFrame:
    """Evaluate the model using k-hop subgraphs on a given mask and return
    predictions as a DataFrame.
    """
    device = runner.device
    model = runner.model.to(device)

    target_mask = getattr(data, "all_mask_loss")
    target_nodes = target_mask.nonzero(as_tuple=True)[0].tolist()

    regression_outs_accumulate = []
    regression_labels_accumulate = []
    node_indices_accumulate = []
    classification_outs_accumulate = []
    classification_labels_accumulate = []

    for node in tqdm(target_nodes):
        subset, edge_index, mapping, _ = k_hop_subgraph(
            node_idx=node,
            num_hops=k,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )

        sub_x = data.x[subset].to(device)
        sub_y = data.y[subset]
        sub_class_labels = data.class_labels[subset]
        sub_mask = target_mask[subset]

        # forward pass
        with torch.no_grad():
            reg_out, class_out = model(sub_x, edge_index.to(device), sub_mask)

        reg_val = reg_out[mapping].squeeze()
        cls_val = class_out[mapping].squeeze()
        label_val = sub_y[mapping].squeeze()
        class_label_val = sub_class_labels[mapping].squeeze()

        regression_outs_accumulate.append(reg_val.item())
        regression_labels_accumulate.append(label_val.item())
        node_indices_accumulate.append(node)
        classification_outs_accumulate.append(cls_val.item())
        classification_labels_accumulate.append(class_label_val.item())

    regression_outs = torch.tensor(regression_outs_accumulate)
    regression_labels = torch.tensor(regression_labels_accumulate)
    node_indices = torch.tensor(node_indices_accumulate)
    classification_outs = torch.tensor(classification_outs_accumulate)
    classification_labels = torch.tensor(classification_labels_accumulate)

    # ensure shape alignment
    assert (
        regression_outs.shape[0]
        == regression_labels.shape[0]
        == node_indices.shape[0]
        == classification_outs.shape[0]
        == classification_labels.shape[0]
    ), "Mismatch in tensor shapes."

    return pd.DataFrame(
        {
            "node_idx": node_indices.cpu().numpy(),
            "prediction": regression_outs.cpu().numpy(),
            "label": regression_labels.cpu().numpy(),
            "class_logits": classification_outs.cpu().numpy(),
            "class_label": classification_labels.cpu().numpy(),
        }
    )


def get_best_predictions(
    df: pd.DataFrame,
    gene_indices: List[int],
    node_idx_to_gene_id: Dict[int, str],
    gencode_to_symbol: Dict[str, str] = None,
    max_low_genes: int = 1000,
    max_mean_diff: float = 0.5,  # note that this is log2(TPM) space
) -> pd.DataFrame:
    """Get predictions for genes given a certain threshold.
    1. We compute the mean absolute difference between the prediction and the
       label (log2tpm space)
    2. We bin by mean label:
        - high if >=5
        - medium if >= 1 and < 5
        - low if > 0 and < 1
    3. We keep all the genes in high and medium, but cap the number of low
    5. Write the final gene symbols to a file for reference
    """
    if gencode_to_symbol is None:
        gencode_to_symbol = {}

    # filter for gene nodes
    df_genes = df[df["node_idx"].isin(gene_indices)].copy()

    # aggregate by node_idx
    df_agg = df_genes.groupby("node_idx", as_index=False).agg(
        {
            "prediction": "mean",
            "label": "mean",
            "class_logits": "mean",
            "class_label": "mean",
        }
    )

    # map node indices to gene IDs
    df_agg["gene_id"] = df_agg["node_idx"].map(node_idx_to_gene_id)

    # add mean_abs_diff to df_genes
    df_agg["mean_abs_diff"] = (df_agg["prediction"] - df_agg["label"]).abs()

    # only keep genes with mean_abs_diff < max_mean_diff
    df_agg = df_agg[df_agg["mean_abs_diff"] < max_mean_diff]

    # bin genes TPM
    df_agg["tpm_bin"] = df_agg["label"].apply(classify_tpm)
    df_agg = df_agg[df_agg["tpm_bin"].isin(["high", "medium", "low"])]

    # split bins
    df_high = df_agg[df_agg["tpm_bin"] == "high"]
    df_medium = df_agg[df_agg["tpm_bin"] == "medium"]
    df_low = df_agg[df_agg["tpm_bin"] == "low"]

    # put cap on lowly expressed genes by sorting on mean_abs_diff, smallest
    # first
    if len(df_low) > max_low_genes:
        df_low = df_low.sort_values(by="mean_abs_diff", ascending=True).head(
            max_low_genes
        )

    # recombine df
    df_filtered = pd.concat([df_high, df_medium, df_low], ignore_index=True)

    # get topk gene IDs and their corresponding node indices
    df_filtered["gene_symbol"] = df_filtered["gene_id"].apply(
        lambda g: map_symbol(g, gencode_to_symbol=gencode_to_symbol)
    )

    # map node indices to gene symbols
    return df_filtered


def invert_symbol_dict(symbol_to_gencode: dict[str, str]) -> dict[str, str]:
    """Create a dictionary that goes ENSG -> symbol."""
    return {ensg: symbol for symbol, ensg in symbol_to_gencode.items()}


def parse_interpret_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config", type=str, help="Path to experiment config."
    )
    parser.add_argument(
        "--run_number",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Run number to determine seed. Options: 1, 2, 3.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of model directory used for training. This will be where the runs are stored. If not specified, will use the model name from the experiment config. If specified, then overwrite (such as for models with replicate runs)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="Name of trained model checkpoint.",
        default="GAT_best_model.pt",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Number of hops for k-hop subgraph.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        help="Sample name for perturbation experiments.",
    )
    # parser.add_argument(
    #     "--experiment",
    #     type=str,
    #     default="all",
    #     help="Which experiment(s) to run: node_features, connected_components, essential, nonessential, coessential, all.",
    # )
    # parser.add_argument(
    #     "--lethal_file",
    #     type=str,
    #     default="/lethal_genes.txt",
    #     help="Path to lethal_genes.txt for essential gene perturbation.",
    # )
    # parser.add_argument(
    #     "--pos_pairs_file",
    #     type=str,
    #     default="/coessential_pos.txt",
    #     help="Path to positive coessential pairs file.",
    # )
    # parser.add_argument(
    #     "--neg_pairs_file",
    #     type=str,
    #     default="/coessential_neg.txt",
    #     help="Path to negative coessential pairs file.",
    # )
    return parser.parse_args()


def _derive_paths(
    experiment_config: ExperimentConfig,
) -> Tuple[str, str, str, str, str]:
    """Derive the following requireds paths and variables from the experiment
    config:
        - root directory
        - split_name
        - idx_file
        - gene_id_lookup
    """
    if len(experiment_config.tissues) > 1:
        raise ValueError("Only one tissue is supported for perturbation experiments.")

    # get root dir
    root_dir = experiment_config.root_dir

    # get split name
    split_name = _dataset_split_name(
        test_chrs=experiment_config.test_chrs,
        val_chrs=experiment_config.val_chrs,
        tpm_filter=0.5,
        percent_of_samples_filter=0.1,
    )
    split_name += "_rna_seq"

    # get idx file
    experiment_name = experiment_config.experiment_name
    graph_type = experiment_config.graph_type
    experiment_dir = f"{root_dir}/experiments/{experiment_name}/graphs/{split_name}"
    idx_file = f"{experiment_dir}/{experiment_name}_{graph_type}_graph_idxs.pkl"

    # get gencode lookup
    gene_id_lookup = (
        f"{experiment_config.reference_dir}/gencode_to_genesymbol_lookup_table.txt"
    )

    return root_dir, split_name, experiment_name, idx_file, gene_id_lookup


def _create_pyg_data(
    experiment_config: ExperimentConfig,
    outpath: Path,
    split_name: str,
    experiment_name: str,
) -> str:
    """Convert numpy graphs to pyg data object and save if it doesn't already
    exist
    """
    # convert graph_data to pytorch
    # check if file exists
    graph_file = f"{outpath}/{experiment_name}_graph_data.pt"
    if os.path.exists(graph_file):
        return graph_file

    target = "rna_seq"
    positional_encoding = True

    data = GraphToPytorch(
        experiment_config=experiment_config,
        split_name=split_name,
        regression_target=target,
        positional_encoding=positional_encoding,
    ).make_data_object()
    torch.save(data, graph_file)

    return graph_file
