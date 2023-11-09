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
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from graph_to_pytorch import graph_to_pytorch
from perturbation import _device_check
from perturbation import _load_GAT_model_for_inference
from utils import filtered_genes_from_bed
from utils import parse_yaml
from utils import TISSUES_early_testing

negative_direction_eqtls = [
    ["liver", "ENSG00000157322.17", -1.3680225998813775, "chr16", 70117253, 70117254],
    ["aorta", "ENSG00000157322.17", -1.3603281368933724, "chr16", 70117253, 70117254],
    ["hippocampus", "ENSG00000154529.14", -1.2627508295506484, "chr9", 42293423, 42293424],
    ["pancreas", "ENSG00000230521.1", -1.2545929219146132, "chr6", 29875340, 29875567],
    ["lung", "ENSG00000230521.1", -1.187098720385814, "chr6", 29875340, 29875567],
]

positive_direction_eqtls = [
    ["left_ventricle", "ENSG00000181126.13", 1.629405140136749, "chr6", 29986227, 29986299],
    ["aorta", "ENSG00000181126.13", 1.57365496389186, "chr6", 29986227, 29986299],
    ["hippocampus", "ENSG00000267065.2", 1.5457906875965717, "chr17", 76787243, 76792227],
    ["liver", "ENSG00000253301.5", 1.5034768626375365, "chr8", 57205334, 57206689],
    ["pancreas", "ENSG00000197083.11", 1.4950484460600506, "chr5", 150798099, 150802038],
]

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


def _flatten_inference_array(input_list):
    result_list = []

    for arr in input_list:
        if len(arr) > 1:
            mean_value = np.mean(arr)
            result_list.append(mean_value)
        else:
            result_list.append(arr[0])

    return result_list


def _eqtl_to_coords(eqtl, graph_idxs):
    """_summary_ of function"""
    tissue, gene, beta, chrom, start, stop = eqtl
    idx = graph_idxs[f"{gene}_{tissue}"]
    begin = start - 1000000
    end = stop + 1000000
    chr_window_start = f'{chrom}_{begin}'
    chr_window_end = f'{chrom}_{end}'
    return idx, start, stop, chr_window_start, chr_window_end


def _get_subset_of_idx_dictionary(dictionary, chrom):
    """_summary_ of function"""
    subdict = {key: value for key, value in dictionary.items() if chrom in key}
    
    
def _get_idxs_for_sv(dictionary, graph_idxs, start, stop):
    idxs = []
    for key, value in dictionary.items():
        _, begin, _ = key.split("_", 2)
        if begin >= start and begin <= stop:
            idx = graph_idxs[key]
            idxs.append(idx)


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


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tissue",
        "-t",
        type=str,
    )
    args = parser.parse_args()
        
    # get big ugly names out the way!
    graph = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_scaled.pkl"
    graph_idxs = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_idxs.pkl"
    checkpoint_file = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/models/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_GAT_2_256_0.0001_batch32_neighbor_full_targetnoscale_idx_expression_only/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_GAT_2_256_0.0001_batch32_neighbor_full_targetnoscale_idx_expression_only_mse_1.843210432337007.pt"
    positive_coessential_genes="/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/coessential_gencode_named_pos.txt"
    negative_coessential_genes="/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/coessential_gencode_named_neg.txt"
    savedir='/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/coessential'
    
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
        batch_size=2048,
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
                

if __name__ == "__main__":
    main()