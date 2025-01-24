#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Use gradient-based saliency maps to explain feature importance."""


import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore


def compute_gradient_saliency(
    model: torch.nn.Module,
    data: Data,
    device: torch.device,
    gene_indices: torch.Tensor,
    k_hops: int = 2,
    regression_loss_type: str = "smooth_l1",
    alpha: float = 0.95,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute saliency w.r.t. all node features."""
    # initialize saliency map
    accumulated_saliency = torch.zeros_like(data.x, device="cpu")

    # data and model prep
    edge_index = data.edge_index.to(device)
    x_cpu = data.x
    y_cpu = data.y
    class_labels_cpu = data.class_labels

    if gene_indices.dim() == 0:
        gene_indices = gene_indices.unsqueeze(0)

    model.to(device)
    model.eval()

    # process batches and accumulate gradients
    for start_idx in range(0, gene_indices.size(0), batch_size):
        end_idx = start_idx + batch_size
        batch_gene_ids = gene_indices[start_idx:end_idx]

        # extract k-hop subgraph
        node_subgraph, edge_subgraph, mapping, _ = k_hop_subgraph(
            batch_gene_ids,
            num_hops=k_hops,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )

        # indexing
        node_subgraph_cpu = node_subgraph.cpu()

        # copy subgraph to device
        x_sub = x_cpu[node_subgraph_cpu].to(device).detach().clone()
        x_sub.requires_grad = True
        edge_index_sub = edge_subgraph

        mask_sub = torch.zeros(x_sub.size(0), dtype=torch.bool, device=device)
        mask_sub[mapping] = True

        # forward pass
        regression_out_sub, logits_sub = model(
            x=x_sub,
            edge_index=edge_index_sub,
            mask=mask_sub,
        )

        # get output
        regression_masked = regression_out_sub[mapping]
        labels_masked = y_cpu[batch_gene_ids].to(device)

        class_masked = logits_sub[mapping]
        class_labels_masked = class_labels_cpu[batch_gene_ids].float().to(device)

        # compute loss
        if regression_loss_type == "rmse":
            mse = F.mse_loss(regression_masked.squeeze(), labels_masked.squeeze())
            regression_loss = torch.sqrt(mse)
        else:
            regression_loss = F.smooth_l1_loss(
                regression_masked.squeeze(), labels_masked.squeeze()
            )

        classification_loss = F.binary_cross_entropy_with_logits(
            class_masked.squeeze(), class_labels_masked.squeeze()
        )

        loss = alpha * regression_loss + (1 - alpha) * classification_loss

        # backprop
        model.zero_grad()
        loss.backward()

        # accumulate gradients
        grad_sub = x_sub.grad.detach().cpu()
        accumulated_saliency[node_subgraph_cpu] += grad_sub

        # free memory
        del x_sub, grad_sub, regression_out_sub, logits_sub
        torch.cuda.empty_cache()

    return accumulated_saliency
