#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run interpretability experiments on a trained model.
1 - Gradient-based saliency maps
2 - Attention weights, if applicable (only if model is GATv2)
3 - PGExplainer subgraph explanations for top nodes
"""


import os
import pickle
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import add_self_loops  # type: ignore
from torch_geometric.utils import coalesce  # type: ignore
from torch_geometric.utils import contains_self_loops  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.attention_weights import \
    get_attention_weights
from omics_graph_learning.interpret.explainer import build_explainer
from omics_graph_learning.interpret.explainer import generate_explanations
from omics_graph_learning.interpret.interpret_utils import _interpret_setup
from omics_graph_learning.interpret.interpret_utils import combine_masks
from omics_graph_learning.interpret.interpret_utils import \
    get_baseline_predictions_k_hop
from omics_graph_learning.interpret.interpret_utils import get_best_predictions
from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import parse_interpret_args
from omics_graph_learning.interpret.perturb_runner import PerturbRunner
from omics_graph_learning.interpret.saliency import compute_gradient_saliency


def scale_saliency(saliency_map: torch.Tensor) -> torch.Tensor:
    """Scale saliency map to [0, 1]."""
    # percentile scaling
    q_low = torch.quantile(saliency_map, 0.05, dim=0)
    q_high = torch.quantile(saliency_map, 0.95, dim=0)
    return torch.clamp((saliency_map - q_low) / (q_high - q_low + 1e-9), 0, 1)


def compute_per_edge_attention(
    attention_weights: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]],
) -> Dict[Tuple[int, int], float]:
    """Compute the average attention weight per edge across all heads and
    occurrences.
    """
    from collections import defaultdict

    edge_attention_sum = defaultdict(float)
    edge_counts = defaultdict(int)

    # total length = only one per layer
    for batches in tqdm(attention_weights.values(), total=2):
        for edge_index, alpha in batches:
            # compute mean attention across heads for each edge
            mean_alpha = alpha.mean(dim=1)

            # iterate over each edge and its mean attention
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                tgt = edge_index[1, i].item()
                attn = mean_alpha[i].item()

                edge = (src, tgt)
                edge_attention_sum[edge] += attn
                edge_counts[edge] += 1

    return {
        edge: total_attn / edge_counts[edge]
        for edge, total_attn in edge_attention_sum.items()
    }


def get_nodes_for_explaining(
    data: Data,
    runner: PerturbRunner,
    gene_indices: torch.Tensor,
    symbol_to_gencode: Dict[str, str],
    node_idx_to_gene_id: Dict[int, str],
    outpath: str,
    num_nodes: int = 250,
) -> List[int]:
    """Get the top nodes for which to run the explainer. If best_prediction_df
    does not exist, run the code to get the best predictions.
    """
    if not os.path.exists(outpath / "best_predictions.csv"):
        gencode_to_symbol = invert_symbol_dict(symbol_to_gencode)

        # get baseline predictions
        baseline_df = get_baseline_predictions_k_hop(
            data=data,
            runner=runner,
        )

        # get best predictions from model
        print("Getting best predictions...")
        best_prediction_df = get_best_predictions(
            df=baseline_df,
            gene_indices=gene_indices,
            node_idx_to_gene_id=node_idx_to_gene_id,
            gencode_to_symbol=gencode_to_symbol,
        )
    else:
        best_prediction_df = pd.read_csv(outpath / "best_predictions.csv")

    # filter to get 250 best in each tpm_bin
    df_filtered = (
        best_prediction_df.groupby("tpm_bin")
        .apply(lambda x: x.nsmallest(num_nodes, "mean_abs_diff"))
        .reset_index(drop=True)
    )

    return df_filtered["node_idx"].tolist()


def main() -> None:
    """Run experiments!"""
    # parse arguments
    args = parse_interpret_args()

    # load experiment setup
    (
        data,
        device,
        runner,
        node_idx_to_gene_id,
        gene_indices,
        idxs_inv,
        idxs,
        symbol_to_gencode,
        outpath,
    ) = _interpret_setup(args)

    # get all_mask for all nodes
    data = combine_masks(data)

    # saliency maps
    gene_indices = data.all_mask_loss.nonzero(as_tuple=False).squeeze()
    saliency_map = compute_gradient_saliency(
        model=runner.model,
        data=data,
        device=device,
        gene_indices=gene_indices,
    )

    # scale saliency map
    scaled_saliency = scale_saliency(saliency_map)
    torch.save(saliency_map, outpath / "raw_saliency_map.pt")
    torch.save(scaled_saliency, outpath / "scaled_saliency_map.pt")

    # # attention weights for genes
    # raw_attention_weights = get_attention_weights(
    #     original_model=runner.model,
    #     data=data,
    #     mask=data.all_mask_loss,
    #     in_channels=42,
    #     hidden_channels=200,
    #     out_channels=1,
    #     heads=2,
    #     num_layers=2,
    # )
    # avg_attention = compute_per_edge_attention(raw_attention_weights)
    # torch.save(avg_attention, outpath / "attention_weights_genes.pt")

    # # attention weights for all nodes
    # raw_attention_weights_all = get_attention_weights(
    #     original_model=runner.model,
    #     data=data,
    #     mask=data.all_mask,
    #     in_channels=42,
    #     hidden_channels=200,
    #     out_channels=1,
    #     heads=2,
    #     num_layers=2,
    # )
    # avg_attention_all = compute_per_edge_attention(raw_attention_weights_all)
    # torch.save(avg_attention_all, outpath / "attention_weights_all_nodes.pt")

    # explainer
    # we run explainer on the top 250 high/medium/low expression genes (750
    # total) to identify subgraph structures
    # target_genes = get_nodes_for_explaining(
    #     data=data,
    #     runner=runner,
    #     gene_indices=gene_indices,
    #     symbol_to_gencode=symbol_to_gencode,
    #     node_idx_to_gene_id=node_idx_to_gene_id,
    #     outpath=outpath,
    # )

    # if not contains_self_loops(data.edge_index):
    #     data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    # # remove duplicate edges
    # data.edge_index, _ = coalesce(data.edge_index, num_nodes=data.num_nodes)

    # explainer = build_explainer(model=runner.model)
    # explanations = {}
    # for target_index in tqdm(target_genes):
    #     try:
    #         explanation = generate_explanations(
    #             explainer=explainer, data=data, index=target_index
    #         )
    #         explanations[target_index] = explanation

    #         orig_nodes = explanation.orig_nodes
    #         print(f"Original nodes in subgraph for target {target_index}: {orig_nodes}")

    #         global_target = explanation.global_target_node
    #         print(f"Global target node index: {global_target}")
    #     except AssertionError as ae:
    #         print(f"AssertionError for target index {target_index}: {ae}")
    #     except Exception as e:
    #         print(f"Error for target index {target_index}: {e}")

    # with open(outpath / "pgexplainer_explanations.pkl", "wb") as f:
    #     pickle.dump(explanations, f)


if __name__ == "__main__":
    main()
