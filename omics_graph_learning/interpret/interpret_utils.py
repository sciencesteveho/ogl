#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Common utility functions for interpretability and perturbation
experiments.
"""


import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore

from omics_graph_learning.interpret.perturb_runner import PerturbRunner


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
    gencode_to_symbol = load_gencode_lookup(lookup_file)

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
        gencode_to_symbol,
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

    # map node indices to gene IDs
    df_genes["gene_id"] = df_genes["node_idx"].map(node_idx_to_gene_id)

    # get absolute difference (|prediction - label|)
    mean_abs_diffs = (
        df_genes.groupby("gene_id")
        .apply(compute_mean_abs_diff)
        .reset_index(name="mean_abs_diff")
    )

    # add mean_abs_diff to df_genes
    df_genes = df_genes.merge(mean_abs_diffs, on="gene_id", how="left")

    # only keep genes with mean_abs_diff < max_mean_diff
    df_genes = df_genes[df_genes["mean_abs_diff"] < max_mean_diff]

    # bin genes TPM
    df_genes["tpm_bin"] = df_genes["label"].apply(classify_tpm)
    df_genes = df_genes[df_genes["tpm_bin"].isin(["high", "medium", "low"])]

    # split bins
    df_high = df_genes[df_genes["tpm_bin"] == "high"]
    df_medium = df_genes[df_genes["tpm_bin"] == "medium"]
    df_low = df_genes[df_genes["tpm_bin"] == "low"]

    # put cap on lowly expressed genes
    unique_low_genes = df_low["gene_id"].nunique()
    if len(unique_low_genes) > max_low_genes:
        keep_low = np.random.choice(unique_low_genes, max_low_genes, replace=False)
        df_low = df_low[df_low["gene_id"].isin(keep_low)]

    # recombine df
    df_filtered = pd.concat([df_high, df_medium, df_low], ignore_index=True)

    # get topk gene IDs and their corresponding node indices
    df_filtered["gene_symbol"] = df_filtered["gene_id"].apply(map_symbol)

    # map node indices to gene symbols
    return df_filtered
