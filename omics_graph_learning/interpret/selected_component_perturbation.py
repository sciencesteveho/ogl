#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""_summary_ of project"""

from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.interpret_utils import calculate_log2_fold_change
from omics_graph_learning.interpret.perturb_runner import PerturbRunner


class SelectedComponentPerturbation:
    """Given a list of (gene_node, node_to_perturb) pairs:
      1. Extract the k-hop subgraph around the gene_node.
      2. Compute a baseline prediction for that gene_node in the subgraph.
      3. For node_to_perturb, systematically zeroes out each feature index
         from 5 through index 41 and re-runs inference to measure impact on
         expression prediction.

    Attributes
    ----------
    data: Data
        PyG Data object containing the full graph
    device: torch.device
        Device on which to run the model
    runner: PerturbRunner
        Wrapper containing the trained model
    idxs_inv: Dict[int, str]
        Mapping from node index -> gene identifier (or label)
    hops: int
        Number of hops to use for k_hop_subgraph
    mask_attr: str
        Name of the mask attribute on Data (e.g., "all_mask_loss")

    Methods
    -------
    run_perturbations(gene_node_pairs):
        Given a list of (gene, node_to_perturb) tuples, run subgraph extraction,
        baseline computation, and feature-zero perturbations for each pair.
    """

    def __init__(
        self,
        data: Data,
        device: torch.device,
        runner: PerturbRunner,
        idxs_inv: dict,
        hops: int = 2,
        mask_attr: str = "all",
    ) -> None:
        """Initialize the SelectedComponentPerturbation object."""
        self.data = data
        self.device = device
        self.runner = runner
        self.idxs_inv = idxs_inv
        self.hops = hops
        self.mask_attr = mask_attr

    def _build_k_hop_subgraph(self, gene_node: int) -> Data:
        """
        Build a k-hop subgraph around gene_node, returning a new Data object.
        """
        node_subgraph, edge_subgraph, _, _ = k_hop_subgraph(
            node_idx=gene_node,
            num_hops=self.hops,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes,
        )

        sub_x = self.data.x[node_subgraph].clone()

        # ensure loss mask
        mask_name = f"{self.mask_attr}_mask_loss"
        if hasattr(self.data, mask_name):
            global_mask = getattr(self.data, mask_name)
            sub_mask = global_mask[node_subgraph]
        else:
            sub_mask = None

        sub_data = Data(
            x=sub_x,
            edge_index=edge_subgraph,
        )
        sub_data.n_id = node_subgraph  # track the original IDs

        if sub_mask is not None:
            setattr(sub_data, mask_name, sub_mask)

        return sub_data

    def _compute_baseline_prediction(self, sub_data: Data, gene_node: int) -> float:
        """Compute the baseline prediction for the gene_node from the
        subgraph.
        """
        idx_in_subgraph = (sub_data.n_id == gene_node).nonzero(as_tuple=True)[0].item()

        with torch.no_grad():
            out, _ = self.runner.model(
                x=sub_data.x,
                edge_index=sub_data.edge_index,
                mask=getattr(sub_data, f"{self.mask_attr}_mask_loss"),
            )
        return out[idx_in_subgraph].item()

    def _perturb_feature_and_predict(
        self,
        sub_data: Data,
        gene_node: int,
        local_node_to_perturb: int,
        feature_idx: int,
        baseline_prediction: float,
    ) -> float:
        """Given sub_data and the local index of the node we want to perturb,
        zero out feature_idx and compute the new prediction for gene_node.
        Return the log2 fold change relative to baseline_prediction.
        """
        # copy the sub_data.x so we do not modify the original
        perturbed_x = sub_data.x.clone()
        perturbed_x[local_node_to_perturb, feature_idx] = 0.0

        with torch.no_grad():
            out, _ = self.runner.model(
                x=perturbed_x,
                edge_index=sub_data.edge_index,
                mask=getattr(sub_data, f"{self.mask_attr}_mask_loss"),
            )

        idx_in_subgraph = (sub_data.n_id == gene_node).nonzero(as_tuple=True)[0].item()
        new_pred = out[idx_in_subgraph].item()
        return calculate_log2_fold_change(baseline_prediction, new_pred)

    def run_perturbations(
        self,
        gene_node_pairs: List[Tuple[int, int]],
    ) -> Dict[str, Dict[str, Dict[int, float]]]:
        """For each (gene_node, node_to_perturb) in gene_node_pairs:
          1. Build k-hop subgraph for gene_node.
          2. Compute baseline prediction for gene_node.
          3  Find the local index of node_to_perturb inside the subgraph.
          4. Systematically zero out features.
          5) Store results in a nested dictionary.

        Returns:
            results[gene_name][node_to_perturb_name][feature_index] = fc
        """
        results = {}

        for gene_node, node_to_perturb in tqdm(
            gene_node_pairs, desc="Selected feature-perturbation experiments"
        ):
            gene_name = self.idxs_inv.get(gene_node, str(gene_node))
            node_to_perturb_name = self.idxs_inv.get(
                node_to_perturb, str(node_to_perturb)
            )

            # build k-hop subgraph
            sub_data = self._build_k_hop_subgraph(gene_node).to(self.device)

            # find the local index of node_to_perturb inside the subgraph
            loc_idxs = (sub_data.n_id == node_to_perturb).nonzero(as_tuple=True)
            if loc_idxs.numel() == 0:
                continue
            local_node_to_perturb = loc_idxs[0].item()

            # compute baseline prediction
            baseline_prediction = self._compute_baseline_prediction(sub_data, gene_node)

            # initialize results dict
            if gene_name not in results:
                results[gene_name] = {}
            if node_to_perturb_name not in results[gene_name]:
                results[gene_name][node_to_perturb_name] = {}

            # systematically zero out features
            max_feat_idx = sub_data.x.shape[1]  # e.g., 42
            for feat_idx in range(5, max_feat_idx):
                fc = self._perturb_feature_and_predict(
                    sub_data,
                    gene_node,
                    local_node_to_perturb,
                    feat_idx,
                    baseline_prediction,
                )
                results[gene_name][node_to_perturb_name][feat_idx] = fc

        return results
