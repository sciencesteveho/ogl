#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run interpretability experiments on a trained model.
1 - Gradient-based saliency maps
2 - Attention weights, if applicable (only if model is GATv2)
3 - PGExplainer subgraph explanations for top nodes
"""


from typing import Dict, List, Tuple

import torch

from omics_graph_learning.interpret.attention_weights import get_attention_weights
from omics_graph_learning.interpret.explainer import build_explainer
from omics_graph_learning.interpret.explainer import train_explainer
from omics_graph_learning.interpret.interpret_utils import _interpret_setup
from omics_graph_learning.interpret.interpret_utils import combine_masks
from omics_graph_learning.interpret.interpret_utils import parse_interpret_args
from omics_graph_learning.interpret.saliency import compute_gradient_saliency


def scale_saliency(saliency_map: torch.Tensor) -> torch.Tensor:
    """Scale saliency map to [0, 1]."""
    saliency_min, _ = saliency_map.min(dim=0, keepdim=True)  # min feature
    saliency_max, _ = saliency_map.max(dim=0, keepdim=True)  # max feature
    return (saliency_map - saliency_min) / (saliency_max - saliency_min + 1e-9)


def compute_per_edge_attention(
    attention_weights: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]],
) -> Dict[Tuple[int, int], float]:
    """Compute the average attention weight per edge across all heads and
    occurrences.
    """
    from collections import defaultdict

    edge_attention_sum = defaultdict(float)
    edge_counts = defaultdict(int)

    for batches in attention_weights.values():
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


def main() -> None:
    """Run experiments!"""
    # parse arguments
    args = parse_interpret_args()

    # load experiment setup
    (
        data,
        device,
        runner,
        _,
        _,
        _,
        _,
        _,
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
    torch.save(scaled_saliency, outpath / "scaled_saliency_map.pt")

    # attention weights for genes
    raw_attention_weights = get_attention_weights(
        original_model=runner.model,
        data=data,
        mask=data.all_mask_loss,
        in_channels=42,
        hidden_channels=200,
        out_channels=1,
        heads=2,
        num_layers=2,
    )
    avg_attention = compute_per_edge_attention(raw_attention_weights)
    torch.save(avg_attention, outpath / "attention_weights_genes.pt")

    # attention weights for all nodes
    raw_attention_weights_all = get_attention_weights(
        original_model=runner.model,
        data=data,
        mask=data.all_mask,
        in_channels=42,
        hidden_channels=200,
        out_channels=1,
        heads=2,
        num_layers=2,
    )
    avg_attention_all = compute_per_edge_attention(raw_attention_weights_all)
    torch.save(avg_attention_all, outpath / "attention_weights_all_nodes.pt")

    # explainer
    pg_explainer = build_explainer(model=runner.model, epochs=30, lr=0.003)
    explanations = train_explainer(pg_explainer, data)
    torch.save(explanations, outpath / "pgexplainer_explanations.pt")


if __name__ == "__main__":
    main()
