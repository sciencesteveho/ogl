#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Use gradient-based saliency maps to explain feature importance."""


import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch_geometric.data import Data  # type: ignore


def compute_gradient_saliency(
    model: torch.nn.Module,
    data: Data,
    device: torch.device,
    mask: torch.Tensor,
    regression_loss_type: str = "smooth_l1",
    alpha: float = 0.95,
    batch_size: int = 256,
) -> torch.Tensor:
    """Compute gradient-based saliency for node features."""
    model.eval()  # safety measure
    data = data.to(device)  # safety measure
    data.x.requires_grad = True  # track gradients on node features

    # initialize empty tensor to accumulate saliency
    saliency_map = torch.zeros_like(data.x)

    masked_indices = mask.nonzero(as_tuple=False).squeeze()

    for start_idx in range(0, masked_indices.size(0), batch_size):
        end_idx = start_idx + batch_size
        batch_indices = masked_indices[start_idx:end_idx]

        # batch mask
        batch_mask = torch.zeros_like(mask, dtype=torch.bool)
        batch_mask[batch_indices] = 1

        # clear gradients
        if data.x.grad is not None:
            data.x.grad.zero_()

        # compute saliency
        regression_out, logits = model(
            x=data.x, edge_index=data.edge_index, mask=batch_mask
        )
        regression_masked = regression_out[batch_mask]
        labels_masked = data.y[batch_mask]

        classification_masked = logits[batch_mask]
        class_labels_masked = data.class_labels[batch_mask].float()

        # compute loss
        if batch_mask.sum() == 0:
            regression_loss = torch.tensor(0.0, device=device)
        elif regression_loss_type == "rmse":
            mse = F.mse_loss(regression_masked.squeeze(), labels_masked.squeeze())
            regression_loss = torch.sqrt(mse)
        elif regression_loss_type == "smooth_l1":
            regression_loss = F.smooth_l1_loss(
                regression_masked.squeeze(), labels_masked.squeeze()
            )
        else:
            raise ValueError("regression_loss_type must be 'rmse' or 'smooth_l1'.")

        # classification part of custom loss
        if batch_mask.sum() == 0:
            classification_loss = torch.tensor(0.0, device=device)
        else:
            classification_loss = F.binary_cross_entropy_with_logits(
                classification_masked.squeeze(),
                class_labels_masked.squeeze(),
            )

        # combine losses
        loss = alpha * regression_loss + (1 - alpha) * classification_loss

        # backpropagation
        model.zero_grad()
        loss.backward()

        # accumulate gradients
        saliency_map[batch_mask] = data.x.grad[batch_mask].detach().cpu()

    data.x.requires_grad = False
    return saliency_map
