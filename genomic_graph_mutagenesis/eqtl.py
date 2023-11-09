#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO
#
# cut -f 1,2,6,7,10,11,12 Jasmine_final_regenotyped_GTEx_SV_eQTLs_all_associations.tsv
# cut -f 1,2,6,7,10,11,12 Jasmine_final_regenotyped_GTEx_SV_eQTLs_all_associations.tsv  | sort -k4,4nr | grep del | grep Lung | head -n 10

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
    ["hippocampus", "ENSG00000178404.9", "8_0_86049_del", -1.2153434357979227, "chr17", 79528294, 79528389],
    ["liver", "ENSG00000117226.11", "0_0_2275_del", -1.0632889503119334, "chr1", 89010225, 89012942],
    ["aorta", "ENSG00000117226.11", "0_0_2275_del", -1.215529755868427, "chr1", 89010225, 89012942],
]

positive_direction_eqtls = [
    ["hippocampus", "ENSG00000267065.2", "2_0_85115_del", 1.5457906875965717, "chr17", 76787243, 76792227],
    ["lung", "ENSG00000214401.4", "12_0_84023_del", 1.2476457467323692, "chr17", 46009357, 46009596],
    ["skeletal_muscle", "ENSG00000100191.5", "1_0_87414_del", 1.4712924976823514, "chr22", 32402561, 32402702],
]


@torch.no_grad()
def inference(model, device, data_loader):
    model.eval()

    pbar = tqdm(total=len(data_loader))

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


def _eqtl_to_coords(eqtl, graph_idxs):
    """_summary_ of function"""
    tissue, gene, sv, beta, chrom, start, stop = eqtl
    idx = graph_idxs[f"{gene}_{tissue}"]
    begin = start - 1000000
    end = stop + 1000000
    return idx, start, stop, begin, end


def _get_subset_of_idx_dictionary(dictionary, chrom, tissue):
    """_summary_ of function"""
    return {key: value for key, value in dictionary.items() if chrom in key and tissue in key}

      
def _get_idxs_for_sv(dictionary, graph_idxs, start, stop):
    idxs = []
    for key, value in dictionary.items():
        _, begin, _ = key.split("_", 2)
        begin = int(begin)
        if begin >= start and begin <= stop:
            idx = graph_idxs[key]
            idxs.append(idx)
    return idxs


def _average_values_across_dict(dictionary):
    """Given a dictionary where each value is a list of floats, returns the same
    dictionary but replaces the list of floats with the average"""
    for key, value in dictionary.items():
        dictionary[key] = np.mean(value)
    return dictionary


def main() -> None:
    """Main function"""
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--tissue",
    #     "-t",
    #     type=str,
    # )
    # args = parser.parse_args()
        
    # get big ugly names out the way!
    graph = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/curated/graphs/curated_full_graph_scaled.pkl"
    graph_idxs = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/curated/graphs/curated_full_graph_idxs.pkl"
    checkpoint_file = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/models/curated_GAT_2_500_0.0001_batch32_neighbor_full_idx_dropout_scaled_expression_only/curated_GAT_2_500_0.0001_batch32_neighbor_full_idx_dropout_scaled_expression_only_mse_1.8369761025042963.pt"
    savedir='/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/'
    
    # parse yaml for params, used to load data
    config = '/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs/ablation_experiments/curated.yaml'
    # parser = argparse.ArgumentParser()
    params = parse_yaml(config)
    
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
        embedding_size=500,
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
    
    def _tensor_out_to_array(tensor, idx):
        return np.stack([x[idx].cpu().numpy() for x in tensor], axis=0)
    
    # get test data cuz we here
    batch_size=32
    test_loader = NeighborLoader(
        data,
        num_neighbors=[5, 5, 5, 5, 5, 3],
        batch_size=batch_size,
        input_nodes=data.test_mask,
    )
    _, outs, labels = inference(
        model=model,
        device=device,
        data_loader=test_loader,
    )
    predictions_median = _tensor_out_to_array(outs, 0)
    labels_median = _tensor_out_to_array(labels, 0)
    with open(f'{savedir}/median_predictions.pkl', 'wb') as file:
        pickle.dump(predictions_median, file)
    with open(f'{savedir}/median_labels.pkl', 'wb') as file:
        pickle.dump(labels_median, file)

    # set up loaders for inference
    batch_size=32
    all_loader = NeighborLoader(
        data,
        num_neighbors=[5, 5, 5, 5, 5, 3],
        batch_size=2048,
        input_nodes=data.all_mask,
    )
    
    # perform inference
    _, outs, labels, expression = all_inference(
        model=model,
        device=device,
        data_loader=all_loader,
    )
    
    # average values across the dictionary
    # each key is an idx
    baseline_expression = _average_values_across_dict(expression)
    
    
    # try qtls
    qtl_dict = {}
    for samples in [negative_direction_eqtls, positive_direction_eqtls]:
        for qtl in samples:
            savename = f'{qtl[0]}_{qtl[1]}_{qtl[2]}_{qtl[3]}'
            idx, start, stop, chr_window_start, chr_window_end = _eqtl_to_coords(qtl, graph_idxs)
            chr_dict = _get_subset_of_idx_dictionary(graph_idxs, chrom=qtl[4], tissue=qtl[0])
            qtl_del_idxs = _get_idxs_for_sv(chr_dict, graph_idxs, chr_window_start, chr_window_end)

            # store baseline
            qtl_dict[savename] = []
            qtl_dict[savename].append(baseline_expression[idx])

            data = graph_to_pytorch(
                experiment_name=params["experiment_name"],
                graph_type='full',
                root_dir=root_dir,
                targets_types=params["training_targets"]["targets_types"],
                test_chrs=params["training_targets"]["test_chrs"],
                val_chrs=params["training_targets"]["val_chrs"],
                node_remove_edges=[qtl_del_idxs],
            )
            perturb_loader = NeighborLoader(
                data,
                num_neighbors=[5, 5, 5, 5, 5, 3],
                batch_size=batch_size,
                input_nodes=data.all_mask,
            )
            _, _, _, perturbed_expression = all_inference(
                model=model,
                device=device,
                data_loader=perturb_loader,
            )
            perturbed_expression = _average_values_across_dict(perturbed_expression)
            qtl_dict[savename].append(perturbed_expression[idx])
            
    # try other tissues
    other_tissues = {}
    
    qtl = positive_direction_eqtls[1]
    real_tissue = qtl[0]
    for tissue in ['aorta', 'hippocampus', 'left_ventricle', 'pancreas', 'liver', 'lung', 'skeletal_muscle', 'small_intestine',]:
        if tissue != real_tissue:
            qtl[0] = tissue
            savename = f'{qtl[0]}_{qtl[1]}_{qtl[2]}_{qtl[3]}'
            try:
                idx, start, stop, chr_window_start, chr_window_end = _eqtl_to_coords(qtl, graph_idxs)
                chr_dict = _get_subset_of_idx_dictionary(graph_idxs, chrom=qtl[4], tissue=tissue)
                qtl_del_idxs = _get_idxs_for_sv(chr_dict, graph_idxs, chr_window_start, chr_window_end)

                # store baseline
                other_tissues[savename] = []
                other_tissues[savename].append(baseline_expression[idx])

                data = graph_to_pytorch(
                    experiment_name=params["experiment_name"],
                    graph_type='full',
                    root_dir=root_dir,
                    targets_types=params["training_targets"]["targets_types"],
                    test_chrs=params["training_targets"]["test_chrs"],
                    val_chrs=params["training_targets"]["val_chrs"],
                    node_remove_edges=[qtl_del_idxs],
                )
                perturb_loader = NeighborLoader(
                    data,
                    num_neighbors=[5, 5, 5, 5, 5, 3],
                    batch_size=batch_size,
                    input_nodes=data.all_mask,
                )
                _, _, _, perturbed_expression = all_inference(
                    model=model,
                    device=device,
                    data_loader=perturb_loader,
                )
                perturbed_expression = _average_values_across_dict(perturbed_expression)
                other_tissues[savename].append(perturbed_expression[idx])
            except KeyError:
                pass
        

if __name__ == "__main__":
    main()
    
"""
Two where it works!
{'hippocampus_ENSG00000178404.9_8_0_86049_del_-1.2153434357979227': [-0.3131829, -0.54981804],
 'lung_ENSG00000214401.4_12_0_84023_del_1.2476457467323692': [1.2725096, 1.475115]}
"""