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

import argparse
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

from graph_to_pytorch import graph_to_pytorch
from perturbation import _device_check
from perturbation import _load_GAT_model_for_inference
from perturbation import _tensor_out_to_array
from utils import filtered_genes_from_bed
from utils import parse_yaml
from utils import TISSUES_early_testing


@torch.no_grad()
def all_inference(model, device, data_loader):
    model.eval()

    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Performing inference")

    mse, outs, labels = [], [], []
    expression = {}
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        # calculate loss
        outs.extend(out[data.all_mask])
        labels.extend(data.y[data.all_mask])
        mse.append(F.mse_loss(out[data.all_mask], data.y[data.all_mask]).cpu())
        loss = torch.stack(mse)
        
        # get idxs
        array_index_only = data.n_id.cpu().numpy() * data.all_mask.cpu().numpy()
        array_index_only = array_index_only[array_index_only != 0]
        outdata = out[data.all_mask].cpu().detach().numpy()
        for idx, _ in enumerate(array_index_only):
            try:
                expression[array_index_only[idx]].append(outdata[idx][0])
            except KeyError:
                expression[array_index_only[idx]] = []
                expression[array_index_only[idx]].append(outdata[idx][0])
                                                     
        pbar.update(1)

    pbar.close()
    return math.sqrt(float(loss.mean())), outs, labels, expression


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


def _average_values_across_dict(dictionary):
    """Given a dictionary where each value is a list of floats, returns the same
    dictionary but replaces the list of floats with the average"""
    for key, value in dictionary.items():
        dictionary[key] = np.mean(value)
    return dictionary


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
    if del in 4
    
    split eqtls into plus and minus, per tissue
    for eqtl
        convert start and end to coords
        in graph
            delete all nodes between start and end
            run inference
            check direction of effect
        store difference from baseline
    """


def _remove_lethal_genes():
    """_summary_ of function"""


def _remove_random_genes():
    """_summary_ of function"""


# def _ablate_housekeeping_genes():
#     """_summary_ of function"""


# def _ablate_random_genes():
#     """_summary_ of function"""


def main(
    mode: str,
    graph: str,
    graph_idxs: str,
    need_baseline: bool = False,
    feat_perturbation: bool = False,
    coessentiality: bool = False,
) -> None:
    """Main function"""
        
    # get big ugly names out the way!
    graph = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_scaled.pkl"
    graph_idxs = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_idxs.pkl"
    checkpoint_file = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/models/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_GAT_2_256_0.0001_batch32_neighbor_full_targetnoscale_idx_expression_only/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_GAT_2_256_0.0001_batch32_neighbor_full_targetnoscale_idx_expression_only_mse_1.843210432337007.pt"
    positive_coessential_genes="/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/coessential_gencode_named_pos.txt"
    negative_coessential_genes="/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/coessential_gencode_named_neg.txt"
    
    # parse yaml for params, used to load data
    config = '/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs/ablation_experiments/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm.yaml'
    # parser = argparse.ArgumentParser()
    params = parse_yaml(config)
    
    def _perturb_loader(data):
        test_loader = NeighborLoader(
            data,
            num_neighbors=[5, 5, 5, 5, 5, 3],
            batch_size=1024,
            input_nodes=data.test_mask,
        )
    
    # open graph
    with open(graph, "rb") as file:
        graph = pickle.load(file)

    # open idxs
    with open(graph_idxs, "rb") as file:
        graph_idxs = pickle.load(file)
        
    # check for device
    device, map_location = _device_check()

    # initialize model and load checkpoint weights
    model = _load_GAT_model_for_inference(
        in_size=41,
        embedding_size=256,
        num_layers=2,
        checkpoint=checkpoint_file,
        map_location=map_location,
        device=device,
    )

    # test_genes = random.sample(list(coessential_genes.keys()), 10)
    # test_random = random.sample(list(random_co_idxs.keys()), 10)
    working_directory = params["working_directory"]
    root_dir = f"{working_directory}/{params['experiment_name']}"
    
    # load data
    data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type='full',
        root_dir=root_dir,
        targets_types=params["training_targets"]["targets_types"],
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
    )

    # set up loaders for inference
    batch_size=32
    all_loader = NeighborLoader(
        data,
        num_neighbors=[5, 5, 5, 5, 5, 3],
        batch_size=32,
        input_nodes=data.all_mask,
    )
    
    # perform inference for the three splits
    _, outs, labels, expression = all_inference(
        model=model,
        device=device,
        data_loader=all_loader,
    )
    
    # average values across the dictionary
    # each key is an idx
    baseline_expression = _average_values_across_dict(expression)

    # coessentiality
    # prepare IDXs for different perturbations
    coessential_genes = _get_idxs_for_coessential_pairs(
        positive_coessential_genes=positive_coessential_genes,
        negative_coessential_genes=negative_coessential_genes,
        graph_idxs=graph_idxs,
    )
    
    # convert to a list of tuples
    # positive = coessential_genes["positive"]
    # negative = coessential_genes["negative"]
    # for key in negative.keys():
    #     negative[key] = [(key, val) for key, values in negative[key].items() for val in values]
    # for key in positive.keys():
    #     positive[key] = [(key, val) for key, values in positive[key].items() for val in values]
    liver_positive = coessential_genes["liver"]["positive"]
    liver_negative = coessential_genes["liver"]["negative"]
    liver_positive = [(key, val) for key, values in liver_positive.items() for val in values]
    liver_negative = [(key, val) for key, values in liver_negative.items() for val in values]

    random_co_idxs = _random_gene_pairs(
        coessential_genes=coessential_genes,
        graph_idxs=graph_idxs,
    )
    
    def _calculate_difference(baseline_dict, perturbed_dict, key1, key2):
        baseline = baseline_expression[key1]
        perturbed = perturbed_dict[key2]
        return abs(baseline - perturbed)
       
    # Perturb graphs for coessential pairs. Save the difference in expression
    pairs = liver_positive
    liver_perturbed_expression = []
    for x in range(10):
        for tup in pairs:
            if tup[0] in baseline_expression.keys() and tup[1] in baseline_expression.keys():
                data = graph_to_pytorch(
                    experiment_name=params["experiment_name"],
                    graph_type='full',
                    root_dir=root_dir,
                    targets_types=params["training_targets"]["targets_types"],
                    test_chrs=params["training_targets"]["test_chrs"],
                    val_chrs=params["training_targets"]["val_chrs"],
                    node_remove_edges=[tup[0]]
                )
                batch_size=32
                perturb_loader = NeighborLoader(
                    data,
                    num_neighbors=[5, 5, 5, 5, 5, 3],
                    batch_size=32,
                    input_nodes=data.all_mask,
                )
                _, _, _, perturbed_expression = all_inference(
                    model=model,
                    device=device,
                    data_loader=perturb_loader,
                )
                liver_perturbed_expression.append(_calculate_difference(baseline_expression, perturbed_expression, tup[0], tup[1]))
                
    # Perturb graphs for random pairs
                
                
                

if __name__ == "__main__":
    main()