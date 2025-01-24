#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Perturb connected components and measure impact on model output."""


import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import subgraph  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.interpret_utils import calculate_log2_fold_change
from omics_graph_learning.interpret.perturb_runner import PerturbRunner


class ConnectedComponentPerturbation:
    """Class to handle perturbation experiments on connected components. Runs
    perturbations on given a model, graph, and gene nodes of interest. Genes
    should be extracted from interpret_utils.get_best_predictions().

    Attributes:
        data: PyG Data object containing the graph
        device: Device to run the model on.
        runner: PerturbRunner object containing the loaded model.
        idxs_inv: Mapping from node index: gene identifier.
        mask_attr: Mask attribute name.

    Methods
    --------
    run_perturbations: run connected component perturbation experiments

    Examples:
    --------
    # Initialize perturbation object
    >>> experiment = ConnectedComponentPerturbation(
            data=data,
            device=device,
            runner=runner,
            idxs_inv=idxs_inv,
            mask_attr="all",
        )

    # Run perturbation experiments
    >>> perturbations = experiment.run_perturbations(
            genes_to_analyze=genes_to_analyze,
        )
    """

    element_types = [
        ("_enhancer", "enhancers"),
        ("_promoter", "promoters"),
        ("_dyadic", "dyadics"),
    ]

    def __init__(
        self,
        data: Data,
        device: torch.device,
        runner: PerturbRunner,
        idxs_inv: Dict[int, str],
        mask_attr: str = "all",
    ) -> None:
        """Initialize the ConnectedComponentPerturbation object."""
        self.data = data
        self.device = device
        self.runner = runner
        self.idxs_inv = idxs_inv
        self.mask_attr = mask_attr

    def _create_subgraph_loader(
        self,
        gene_node: int,
        num_hops: int,
        batch_size: int = 1,
    ) -> NeighborLoader:
        """Create a NeighborLoader to fetches subgraph around a specific gene
        node. We take avg_edges * 2 neighbors per hop to ensure we don't miss
        critical parts of the graph.
        """
        return NeighborLoader(
            data=self.data,
            num_neighbors=[self.data.avg_edges * 2] * num_hops,
            batch_size=batch_size,
            input_nodes=torch.tensor([gene_node], dtype=torch.long),
            shuffle=False,
        )

    def _compute_baseline_prediction(
        self,
        sub_data: Data,
        gene_node: int,
    ) -> float:
        """Compute the baseline prediction for a given gene node in the
        subgraph.

        Args:
            runner: PerturbRunner object containing the loaded model
            sub_data: Data subgraph batch
            gene_node: idx of the gene of interest
            mask_attr: Attribute name for the mask

        Returns:
            float: The baseline prediction for the gene node.
        """
        idx_in_subgraph = (sub_data.n_id == gene_node).nonzero(as_tuple=True)[0].item()
        mask_tensor = getattr(sub_data, f"{self.mask_attr}_mask_loss")

        with torch.no_grad():
            baseline_out, _ = self.runner.model(
                x=sub_data.x,
                edge_index=sub_data.edge_index,
                mask=mask_tensor,
            )
        return baseline_out[idx_in_subgraph].item()

    def _get_elements_in_subgraph(
        self,
        sub_data: Data,
        regulatory_element: str,
    ) -> List[int]:
        """Return the global node IDs for regulatory elements present in
        sub_data.
        """
        regulatory_elements = []
        for local_i in range(sub_data.num_nodes):
            global_id = sub_data.n_id[local_i].item()
            node_name = self.idxs_inv.get(global_id, "")
            if regulatory_element in node_name:
                regulatory_elements.append(global_id)
        return regulatory_elements

    def _remove_nodes_and_predict(
        self,
        sub_data: Data,
        nodes_to_remove: List[int],
        gene_node: int,
    ) -> Optional[float]:
        """Remove the specified nodes from the subgraph, then compute the
        model's prediction for the gene node in the perturbed subgraph.

        Args:
            runner: PerturbRunner object containing the loaded model.
            sub_data: Subgraph batch.
            node_to_remove: Node to remove from the subgraph.
            gene_node: Gene node to predict.
            device: Device to run the model on.
            mask_attr: Attribute name for the mask.

        Returns:
            Optional[float]: The perturbation prediction for the gene node, or
            None if the gene_node is missing in the perturbed subgraph.
        """
        if not nodes_to_remove:
            return None

        mask_tensor = getattr(sub_data, f"{self.mask_attr}_mask_loss")

        # find nodes to remove
        remove_local_idxs = set()
        for global_id in nodes_to_remove:
            loc = (sub_data.n_id == global_id).nonzero(as_tuple=True)[0]
            if len(loc) > 0:
                remove_local_idxs.add(loc.item())

        if not remove_local_idxs:
            return None

        keep_mask = torch.tensor(
            [i not in remove_local_idxs for i in range(sub_data.num_nodes)],
            dtype=torch.bool,
            device=self.device,
        )

        # create perturbed subgraph
        perturbed_edge_idx, _, _ = subgraph(
            subset=keep_mask,
            edge_index=sub_data.edge_index,
            relabel_nodes=True,
            num_nodes=sub_data.num_nodes,
            return_edge_mask=True,
        )

        perturbed_x = sub_data.x[keep_mask]
        perturbed_mask = mask_tensor[keep_mask]
        perturbed_n_id = sub_data.n_id[keep_mask]

        # ensure gene_node has not been removed
        if (perturbed_n_id == gene_node).sum() == 0:
            return None

        idx_in_perturbed = (
            (perturbed_n_id == gene_node).nonzero(as_tuple=True)[0].item()
        )

        # inference
        with torch.no_grad():
            perturbed_out, _ = self.runner.model(
                x=perturbed_x,
                edge_index=perturbed_edge_idx,
                mask=perturbed_mask,
            )
        return perturbed_out[idx_in_perturbed].item()

    def _perform_single_node_perturbations(
        self,
        sub_data: Data,
        gene_node: int,
        selected_nodes: List[int],
        baseline_prediction: float,
        hop_dist_map: Dict[int, int],
        store_dict: Dict[str, Dict[str, Any]],
    ) -> None:
        """Remove nodes in selected_nodes one at a time, compute fold_change,
        hop distance, and store in a dictionary.
        """
        for node_remove in selected_nodes:
            new_pred = self._remove_nodes_and_predict(
                sub_data, [node_remove], gene_node
            )
            if new_pred is None:
                continue

            node_name = self.idxs_inv.get(node_remove, str(node_remove))
            store_dict[node_name] = {
                "fold_change": calculate_log2_fold_change(
                    baseline_prediction, new_pred
                ),
                "hop_distance": hop_dist_map.get(node_remove, -1),
            }

    def _perform_grouped_node_removal(
        self,
        sub_data: Data,
        gene_node: int,
        baseline_prediction: float,
        element_ids: List[int],
        group_size: int,
    ) -> Optional[Tuple[Tuple[str, ...], float]]:
        """Randomly removes `group_size` nodes from `element_ids` from the
        subgraph.

        Returns None if subgraph can't be formed or gene_node is lost.
        """
        chosen = random.sample(element_ids, group_size)
        new_pred = self._remove_nodes_and_predict(
            sub_data=sub_data,
            nodes_to_remove=chosen,
            gene_node=gene_node,
        )

        if new_pred is None:
            return None

        chosen_names = tuple(sorted(self.idxs_inv.get(eid, str(eid)) for eid in chosen))
        return (chosen_names, calculate_log2_fold_change(baseline_prediction, new_pred))

    def _perform_joint_re_perturbations(
        self,
        sub_data: Data,
        gene_node: int,
        baseline_prediction: float,
        element_type: str,
        group_size: int,
        store_dict: Dict[Tuple[str, ...], float],
    ) -> None:
        """Do random removals of elements of type `element_type`."""
        element_ids = self._get_elements_in_subgraph(
            sub_data=sub_data, regulatory_element=element_type
        )
        if len(element_ids) < 3:
            return

        result = self._perform_grouped_node_removal(
            sub_data=sub_data,
            gene_node=gene_node,
            baseline_prediction=baseline_prediction,
            element_ids=element_ids,
            group_size=group_size,
        )
        if result is not None:
            (names, fc) = result
            store_dict[names] = fc

    def run_perturbations(
        self,
        genes_to_analyze: List[int],
        num_hops: int = 5,
        max_nodes_to_perturb: int = 150,
    ) -> Dict[str, Any]:
        """
        For each gene in genes_to_analyze:
          1) build subgraph
          2) compute baseline
          3) single-node perturbations
          4) random double/triple removal for enhancers, promoters, dyadics,
             etc.

        Returns:
            (dict): pertubations
        """
        perturbations = {
            "single": {},
            "double_enhancers": {},
            "triple_enhancers": {},
            "double_promoters": {},
            "triple_promoters": {},
            "double_dyadics": {},
            "triple_dyadics": {},
        }

        for gene_node in tqdm(
            genes_to_analyze, desc="Connected component pertubration experiments"
        ):

            # subgraph loader
            loader = self._create_subgraph_loader(
                gene_node=gene_node, num_hops=num_hops
            )
            sub_data = next(iter(loader)).to(self.device)

            # compute baseline
            baseline = self._compute_baseline_prediction(
                sub_data=sub_data, gene_node=gene_node
            )

            # get nodes to perturb
            selected_nodes = self._get_nodes_to_perturb(
                sub_data=sub_data,
                gene_node=gene_node,
                max_nodes_to_perturb=max_nodes_to_perturb,
            )
            if not selected_nodes:
                continue

            # compute hop distances
            hop_map = self._compute_hop_distances(
                sub_data=sub_data, gene_node=gene_node
            )

            # run single node perturbations
            self._perform_single_node_perturbations(
                sub_data=sub_data,
                gene_node=gene_node,
                selected_nodes=selected_nodes,
                baseline_prediction=baseline,
                hop_dist_map=hop_map,
                store_dict=perturbations["single"],
            )

            # run joint node perturbations
            for element_string, element in self.element_types:
                # double removal
                self._perform_joint_re_perturbations(
                    sub_data=sub_data,
                    gene_node=gene_node,
                    baseline_prediction=baseline,
                    element_type=element_string,
                    group_size=2,
                    store_dict=perturbations[f"double_{element}"],
                )
                # triple removal
                self._perform_joint_re_perturbations(
                    sub_data=sub_data,
                    gene_node=gene_node,
                    baseline_prediction=baseline,
                    element_type=element_string,
                    group_size=3,
                    store_dict=perturbations[f"triple_{element}"],
                )

        return perturbations

    @staticmethod
    def _get_nodes_to_perturb(
        sub_data: Data,
        gene_node: int,
        max_nodes_to_perturb: int,
    ) -> List[int]:
        """Get list of candidate nodes to remove (excludes the gene_node itself). If
        more nodes are specified than are available, randomly sample from the
        available nodes.

        Args:
            sub_data: Data subgraph batch
            gene_node: idx of the gene of interest
            max_nodes_to_perturb: Max number of nodes to remove at once

        Returns:
            List of node indices to remove
        """
        # exclude the gene node itself
        nodes_to_perturb = sub_data.n_id[sub_data.n_id != gene_node]

        if len(nodes_to_perturb) == 0:
            return []

        return (
            random.sample(nodes_to_perturb.tolist(), max_nodes_to_perturb)
            if len(nodes_to_perturb) > max_nodes_to_perturb
            else nodes_to_perturb.tolist()
        )

    @staticmethod
    def _compute_hop_distances(sub_data: Data, gene_node: int) -> Dict[int, int]:
        """Compute hop distances from the gene_node to every other node in the
        subgraph.

        Returns:
            {node: hop_distance from gene_node}
        """
        # convert to CPU if necessary
        sub_data_cpu = sub_data.clone().cpu()

        # construct a networkx subgraph
        subgraph_nx = to_networkx(sub_data_cpu, to_undirected=True)
        mapping_nx = {
            i: sub_data_cpu.n_id[i].item() for i in range(sub_data_cpu.num_nodes)
        }
        subgraph_nx = nx.relabel_nodes(subgraph_nx, mapping_nx)

        return nx.single_source_shortest_path_length(subgraph_nx, gene_node)
