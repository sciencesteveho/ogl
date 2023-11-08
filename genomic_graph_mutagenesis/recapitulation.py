#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO
#
# Load model
# Load data w/ modification
# loader = NeighborLoader(new_data, input_nodes=new_node_ids, ...)
# out = model(new_data.x, new_data.edge_index)
# compare co-essential perturbations to random perturbation


"""_summary_ of project"""

import csv
import math
import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from gnn import GATv2
from graph_to_pytorch import graph_to_pytorch
from utils import filtered_genes_from_bed
from utils import TISSUES_early_testing


def _load_model_for_inference():
    """_summary_ of function"""
    model = GATv2(
        in_size=41,
        embedding_size=256,
        out_channels=1,
        num_layers=2,
        heads=2,
    ).to_device()

    return model


def _load_checkpoint(checkpoint: str, map_location: str):
    checkpoint = torch.load(checkpoint, map_location=map_location)


def _tensor_out_to_array(tensor, idx):
    return np.stack([x[idx].cpu().numpy() for x in tensor], axis=0)


def _pre_nest_tissue_dict(tissues):
    return {tissue: {} for tissue in tissues}


def _get_idxs_for_coessential_pairs(
    positive_coessential_genes: str,
    negative_coessential_genes: str,
    graph_idxs: Dict[str, str],
) -> List[Tuple[int, int]]:
    """_summary_ of function
    
    Create a dictionary of coessential genes for each tissue of the following
    format:
        coessential_genes: {
            'tissue_1': {
                positive: {idx1: [], idx2: []},
                negative: {idx1: [], idx2: []},
            },
            'tissue_2': {
                positive: {idx1: [], idx2: []},
                negative: {idx1: [], idx2: []},
            },
        }
    """
    def _prepopulate_dict_with_keys(pairs, tissue):
        all_genes = [tup[0] for tup in pairs] + [tup[1] for tup in pairs]
        return {graph_idxs[f"{gene}_{tissue}"]: [] for gene in all_genes if f"{gene}_{tissue}" in graph_idxs.keys()}
        
    def _populate_dict_with_co_pairs(pairs, tissue, pre_dict):    
        for pair in pairs:
            try:
                idx1 = graph_idxs[f"{pair[0]}_{tissue}"]
                idx2 = graph_idxs[f"{pair[1]}_{tissue}"]
                if idx2 < idx1:
                    pre_dict[idx1].append(idx2)
            except KeyError:
                pass
        return {key: value for key, value in pre_dict.items() if value}

    positive_pairs = [
        (line[0], line[1])
        for line in csv.reader(open(positive_coessential_genes, newline=""), delimiter="\t")
    ]

    negative_pairs = [
        (line[0], line[1])
        for line in csv.reader(open(negative_coessential_genes, newline=""), delimiter="\t")
    ]
    
    coessential_genes = _pre_nest_tissue_dict(TISSUES_early_testing)
    for key in coessential_genes.keys():
        coessential_genes[key] = {
            "positive": _populate_dict_with_co_pairs(pairs=positive_pairs, tissue=key, pre_dict=_prepopulate_dict_with_keys(pairs=positive_pairs, tissue=key),),
            "negative": _populate_dict_with_co_pairs(pairs=negative_pairs, tissue=key, pre_dict=_prepopulate_dict_with_keys(pairs=negative_pairs, tissue=key),),
        }
    
    return coessential_genes


def _random_gene_pairs(
    coessential_genes: Dict[int, List[int]],
    graph_idxs: Dict[str, str],
    model_dir: str = '/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/alldata_combinedloops',
) -> List[Tuple[int, int]]:
    """_summary_ of function"""
    
    def _get_number_of_pairs(subdict):
        total = sum(len(value) + 1 for value in subdict.values())
        return total

    def _gencode_to_idx_list(tissue, genes):
        gencode_keys = set(f"{gene}_{tissue}" for gene in genes)
        return [graph_idxs[key] for key in gencode_keys if key in graph_idxs]

    def _generate_random_pairs(tissue_genes, total_pairs):
        gencode_idxs = _gencode_to_idx_list(tissue=tissue, genes=tissue_genes)
        random_pairs = []

        for _ in range(total_pairs):
            idx1 = random.choice(gencode_idxs)
            idx2 = random.choice(gencode_idxs)
            if idx2 < idx1:
                random_pairs.append((idx1, idx2))

        return random_pairs
    
    # initialize the dict
    random_copairs = _pre_nest_tissue_dict(TISSUES_early_testing)
    
    # get number of pairs to emulate, use first tissue as reference
    first_tissue = list(coessential_genes.keys())[0]
    total_positive_pairs = _get_number_of_pairs(coessential_genes[first_tissue]["positive"])
    total_negative_pairs = _get_number_of_pairs(coessential_genes[first_tissue]["negative"])

    for tissue, subdict in random_copairs.items():
        subdict["positive"] = {}
        subdict["negative"] = {}
        tissue_genes = filtered_genes_from_bed(
            tpm_filtered_genes=f"{model_dir}/{tissue}/tpm_filtered_genes.bed",
        )
        positive_pairs = _generate_random_pairs(tissue_genes, total_positive_pairs)
        negative_pairs = _generate_random_pairs(tissue_genes, total_negative_pairs)

        for idx1, idx2 in positive_pairs:
            if idx1 not in subdict["positive"]:
                subdict["positive"][idx1] = []
            subdict["positive"][idx1].append(idx2)
        
        for idx1, idx2 in negative_pairs:
            if idx1 not in subdict["negative"]:
                subdict["negative"][idx1] = []
            subdict["negative"][idx1].append(idx2)

    return random_copairs


def _store_baseline():
    """STEPS
    1. store baseline TPMs for each gene from checkpoint model
    for coessential dict and random pairs dict:
        for each key:
            remove the key from the graph
            run inference
            for each value:
                get inferred TPM
                calculate difference from baseline
                store back into a dict
    """
        
def _scale_coordinate(scaler, coordinate):
    """_summary_ of function"""


def _perturb_eQTLs():
    """
    idxs:
        0 = tissue
        1 = gene
        5 = beta
        8 = chrom
        9 = start
        10 = end
    if del in 4"""


def _remove_lethal_genes():
    """_summary_ of function"""


def _remove_random_genes():
    """_summary_ of function"""


def _ablate_housekeeping_genes():
    """_summary_ of function"""


def _ablate_random_genes():
    """_summary_ of function"""


@torch.no_grad()
def test(model, device, data_loader, epoch):
    model.eval()

    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse, outs, labels = [], [], []
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        # calculate loss
        outs.extend(out[data.test_mask])
        labels.extend(data.y[data.test_mask])
        mse.append(F.mse_loss(out[data.test_mask], data.y[data.test_mask]).cpu())
        loss = torch.stack(mse)

        pbar.update(1)

    pbar.close()
    # print(spearman(torch.stack(outs), torch.stack(labels)))
    return math.sqrt(float(loss.mean())), outs, labels


def main(
    mode: str,
    graph: str,
    graph_idxs: str,
    need_baseline: bool = False,
    feat_perturbation: bool = False,
    coessentiality: bool = False,
) -> None:
    """Main function"""

    def _perturb_loader(data):
        test_loader = NeighborLoader(
            data,
            num_neighbors=[5, 5, 5, 5, 5, 3],
            batch_size=1024,
            input_nodes=data.test_mask,
        )

    # check for device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = torch.device("cuda:" + str(0))
        map_location = torch.device("cuda:" + str(0))
    else:
        device = torch.device("cpu")
        map_location = torch.device("cpu")

    # only using to check size for model init
    # data = graph_to_pytorch(
    #     root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
    #     graph_type="full",
    # )
    # data.x.shape[1]  # 41

    # prepare stuff
    graph = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_scaled.pkl"
    graph_idxs = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_idxs.pkl"

    # open graph
    with open(graph, "rb") as file:
        graph = pickle.load(file)

    # open idxs
    with open(graph_idxs, "rb") as file:
        graph_idxs = pickle.load(file)

    # initialize model model
    model = GraphSAGE(
        in_size=41,
        embedding_size=250,
        out_channels=2,
        num_layers=2,
    ).to(device)

    # load checkpoint
    checkpoint_file = "/ocean/projects/bio210019p/stevesho/data/preprocess/GraphSAGE_2_250_5e-05_batch1024_neighbor_idx_early_epoch_52_mse_0.8029395813403142.pt"
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    # prepare IDXs for different perturbations
    coessential_genes = _get_idxs_for_coessential_pairs(
        positive_coessential_genes="/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/coessential_gencode_named_pos.txt",
        negative_coessential_genes="/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/coessential_gencode_named_neg.txt",
        graph_idxs=graph_idxs,
    )

    random_co_idxs = _random_gene_pairs(
        coessential_genes=coessential_genes,
        graph_idxs=graph_idxs,
    )

    test_genes = random.sample(list(coessential_genes.keys()), 10)
    test_random = random.sample(list(random_co_idxs.keys()), 10)

    if need_baseline:
        # get baseline expression
        baseline_data = graph_to_pytorch(
            root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
            graph_type="full",
            only_expression_no_fold=True,
        )
        loader = NeighborLoader(
            data=baseline_data,
            num_neighbors=[5, 5, 5, 5, 5, 3],
            batch_size=1024,
            input_nodes=baseline_data.test_mask,
        )
        rmse, outs, labels = test(
            model=model,
            device=device,
            data_loader=loader,
            epoch=0,
        )

        predictions_median = _tensor_out_to_array(outs, 0)
        # predictions_fold = _tensor_out_to_array(outs, 1)
        labels_median = _tensor_out_to_array(labels, 0)
        # labels_fold = _tensor_out_to_array(labels, 1)

        with open("base_predictions_median.pkl", "wb") as f:
            pickle.dump(predictions_median, f)

        # with open("predictions_fold.pkl", "wb") as f:
        #     pickle.dump(predictions_fold, f)

        with open("base_labels_median.pkl", "wb") as f:
            pickle.dump(labels_median, f)

        with open("labels_fold.pkl", "wb") as f:
            pickle.dump(labels_fold, f)


    # coessentiality
    # get baseline expression
    if coessentiality:
        baselines = {}
        for gene in coessential_genes.keys():
            baselines[gene] = []
            baseline_data = graph_to_pytorch(
                root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
                graph_type="full",
                single_gene=gene,
            )
            loader = NeighborLoader(
                data=baseline_data,
                num_neighbors=[5, 5, 5, 5, 5, 3],
                batch_size=1024,
                input_nodes=baseline_data.test_mask,
            )
            mse, outs, labels = test(
                model=model,
                device=device,
                data_loader=loader,
                epoch=0,
            )
            baselines.append(baseline)
            # baseline_measure
            for co_gene in coessential_genes[gene]:
                perturbed_graph = graph_to_pytorch(
                    root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
                    graph_type="full",
                    single_gene=gene,
                    node_remove_edges=co_gene,
                )
                loader = _perturb_loader(perturbed_graph)
                inference = test(
                    model=model,
                    device=device,
                    data_loader=loader,
                    epoch=0,
                )
                baselines[gene].append(inference)

        with open("coessentiality.pkl", "wb") as f:
            pickle.dump(baselines, f)


if __name__ == "__main__":
    main()
    # main(
    #     model="/ocean/projects/bio210019p/shared/model_checkpoint.pt",
    #     graph="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/scaled/all_tissue_full_graph_scaled.pkl",
    #     graph_idxs="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/all_tissue_full_graph_idxs.pkl",
    # )

# pos_idxs, neg_idxs = {}, {}
# for tissue in TISSUES_early_testing:
#     for tup in positive_pairs:
#         pos_idxs[graph_idxs[f"{tup[0]}_{tissue}"]] = graph_idxs[f"{tup[1]}_{tissue}"]
#     pos_idxs[]
#     pos_idxs.extend(
#         (graph_idxs[f"{tup[0]}_{tissue}"], graph_idxs[f"{tup[1]}_{tissue}"])
#         for tup in positive_pairs
#     )
#     neg_idxs.extend(
#         (graph_idxs[f"{tup[0]}_{tissue}"], graph_idxs[f"{tup[1]}_{tissue}"])
#         for tup in negative_pairs
#     )
