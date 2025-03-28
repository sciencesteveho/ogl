#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Perturb connected components and measure impact on model output."""


import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx  # type: ignore
import torch
from torch_geometric.data import Batch  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from torch_geometric.utils import subgraph  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.interpret_utils import \
    calculate_log2_fold_change
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
        hops: int = 2,
        mask_attr: str = "all",
    ) -> None:
        """Initialize the ConnectedComponentPerturbation object."""
        self.data = data
        self.device = device
        self.runner = runner
        self.idxs_inv = idxs_inv
        self.mask_attr = mask_attr
        self.hops = hops

    def _build_k_hop_subgraph(
        self,
        gene_node: int,
    ) -> Data:
        """Build a k-hop subgraph around the gene node."""
        node_subgraph, edge_subgraph, mapping, _ = k_hop_subgraph(
            node_idx=gene_node,
            num_hops=self.hops,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes,
        )

        sub_x = self.data.x[node_subgraph]

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

        sub_data.n_id = node_subgraph
        if sub_mask is not None:
            setattr(sub_data, mask_name, sub_mask)

        return sub_data

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

    def _create_perturbed_subgraph(
        self,
        sub_data: Data,
        nodes_to_remove: List[int],
        gene_node: int,
    ) -> Optional[Data]:
        """Given a subgraph, remove the specified nodes and return the resulting
        perturbed subgraph as a new Data object. Returns None if none of the
        nodes to remove are found or if the gene_node is no longer present
        after perturbation.
        """
        if not nodes_to_remove:
            return None

        mask_tensor = getattr(sub_data, f"{self.mask_attr}_mask_loss")
        remove_local_idxs = set()
        for global_id in nodes_to_remove:
            loc = (sub_data.n_id == global_id).nonzero(as_tuple=True)[0]
            if loc.numel() > 0:
                remove_local_idxs.add(loc.item())

        if not remove_local_idxs:
            return None

        keep_mask = torch.tensor(
            [i not in remove_local_idxs for i in range(sub_data.num_nodes)],
            dtype=torch.bool,
            device=self.device,
        )

        # create perturbed subgraph using the keep mask
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

        # ensure the gene_node still exists
        if (perturbed_n_id == gene_node).sum() == 0:
            return None

        mini_data = Data(
            x=perturbed_x,
            edge_index=perturbed_edge_idx,
        )
        mini_data.n_id = perturbed_n_id
        setattr(mini_data, f"{self.mask_attr}_mask_loss", perturbed_mask)

        return mini_data

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
        model's prediction for the gene_node in the perturbed subgraph.
        """
        perturbed_sub = self._create_perturbed_subgraph(
            sub_data, nodes_to_remove, gene_node
        )
        if perturbed_sub is None:
            return None

        idx_in_perturbed = (
            (perturbed_sub.n_id == gene_node).nonzero(as_tuple=True)[0].item()
        )

        with torch.no_grad():
            perturbed_out, _ = self.runner.model(
                x=perturbed_sub.x,
                edge_index=perturbed_sub.edge_index,
                mask=getattr(perturbed_sub, f"{self.mask_attr}_mask_loss"),
            )
        return perturbed_out[idx_in_perturbed].item()

    def _remove_nodes_and_predict_batch(
        self,
        sub_data: Data,
        list_of_nodes_to_remove: List[int],
        gene_node: int,
        # batch_size: int = 8,
        batch_size: int = 6,
    ) -> List[Optional[float]]:
        """For each node in list_of_nodes_to_remove, create a perturbed
        subgraph. Then, process these perturbed subgraphs in mini-batches (of
        size batch_size) to run inference in fewer forward passes. The method
        returns a list of predictions (or None when a perturbed subgraph
        could not be created) in the same order as list_of_nodes_to_remove.
        """
        # build a list of perturbed subgraphs
        data_objs = [
            self._create_perturbed_subgraph(sub_data, [global_id], gene_node)
            for global_id in list_of_nodes_to_remove
        ]

        # build a mapping from original index to valid index
        orig_to_valid = {}
        valid_data_objs = []
        for idx, d in enumerate(data_objs):
            if d is not None:
                orig_to_valid[idx] = len(valid_data_objs)
                valid_data_objs.append(d)

        if not valid_data_objs:
            return [None] * len(list_of_nodes_to_remove)

        predictions_by_valid_idx = {}

        # batch inference
        for start in range(0, len(valid_data_objs), batch_size):
            batches = valid_data_objs[start : start + batch_size]
            batch = Batch.from_data_list(batches).to(self.device)

            with torch.no_grad():
                out, _ = self.runner.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    mask=getattr(batch, f"{self.mask_attr}_mask_loss"),
                )

            for i in range(len(batches)):
                node_mask = batch.batch == i
                batch_n_id = batch.n_id[node_mask]
                loc_gene = (batch_n_id == gene_node).nonzero(as_tuple=True)[0]
                if loc_gene.numel() == 0:
                    predictions_by_valid_idx[start + i] = None
                else:
                    idx_in_subgraph_output = loc_gene.item()
                    pred_value = out[node_mask][idx_in_subgraph_output].item()
                    predictions_by_valid_idx[start + i] = pred_value

        # reconstruct the predictions in the original ordering of
        # list_of_nodes_to_remove
        predictions = []
        for i in range(len(list_of_nodes_to_remove)):
            if i in orig_to_valid:
                valid_idx = orig_to_valid[i]
                predictions.append(predictions_by_valid_idx.get(valid_idx))
            else:
                predictions.append(None)

        return predictions

    def _perform_single_node_perturbations(
        self,
        sub_data: Data,
        gene_node: int,
        gene_name: str,
        selected_nodes: List[int],
        baseline_prediction: float,
        hop_dist_map: Dict[int, int],
        store_dict: Dict[str, Dict[str, Any]],
    ) -> None:
        """Remove nodes in selected_nodes one at a time, compute fold_change,
        hop distance, and store in a dictionary.
        """
        if gene_name not in store_dict:
            store_dict[gene_name] = {}

        predictions = self._remove_nodes_and_predict_batch(
            sub_data, selected_nodes, gene_node
        )

        for node_to_remove, new_pred in zip(selected_nodes, predictions):
            if new_pred is None:
                continue
            node_name = self.idxs_inv.get(node_to_remove, str(node_to_remove))

            store_dict[gene_name][node_name] = {
                "fold_change": calculate_log2_fold_change(
                    baseline_prediction, new_pred
                ),
                "hop_distance": hop_dist_map.get(node_to_remove, -1),
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
        gene_name: str,
        baseline_prediction: float,
        element_type: str,
        group_size: int,
        store_dict: Dict[Tuple[str, ...], float],
    ) -> None:
        """Do random removals of elements of type `element_type`."""
        if gene_name not in store_dict:
            store_dict[gene_name] = {}

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
            store_dict[gene_name][names] = fc

    def run_perturbations(
        self,
        genes_to_analyze: List[int],
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
            # get gene name
            gene_name = self.idxs_inv.get(gene_node, str(gene_node))

            # build deterministic subgraph
            sub_data = self._build_k_hop_subgraph(gene_node=gene_node).to(self.device)

            # compute baseline
            baseline = self._compute_baseline_prediction(
                sub_data=sub_data, gene_node=gene_node
            )

            # get nodes to perturb
            selected_nodes = self._get_nodes_to_perturb(
                sub_data=sub_data,
                gene_node=gene_node,
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
                gene_name=gene_name,
                selected_nodes=selected_nodes,
                baseline_prediction=baseline,
                hop_dist_map=hop_map,
                store_dict=perturbations["single"],
            )

            # only run joint perturbations on 2-hops
            if self.hops == 2:
                for element_string, element in self.element_types:
                    # double removal
                    self._perform_joint_re_perturbations(
                        sub_data=sub_data,
                        gene_node=gene_node,
                        gene_name=gene_name,
                        baseline_prediction=baseline,
                        element_type=element_string,
                        group_size=2,
                        store_dict=perturbations[f"double_{element}"],
                    )
                    # triple removal
                    self._perform_joint_re_perturbations(
                        sub_data=sub_data,
                        gene_node=gene_node,
                        gene_name=gene_name,
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
    ) -> List[int]:
        """Get list of candidate nodes to remove (excludes the gene_node itself)

        Args:
            sub_data: Data subgraph batch
            gene_node: idx of the gene of interest

        Returns:
            List of node indices to remove
        """
        # exclude the gene node itself
        nodes_to_perturb = sub_data.n_id[sub_data.n_id != gene_node]
        return [] if len(nodes_to_perturb) == 0 else nodes_to_perturb.tolist()

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
