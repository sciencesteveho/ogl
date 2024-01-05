#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO
#
# cut -f 1,2,6,7,10,11,12 Jasmine_final_regenotyped_GTEx_SV_eQTLs_all_associations.tsv
# cut -f 1,2,6,7,10,11,12 Jasmine_final_regenotyped_GTEx_SV_eQTLs_all_associations.tsv  | sort -k4,4nr | grep del | grep Lung | head -n 10
# awk '{ seen[$6]++ } seen[$6] >= 2' Jasmine_final_regenotyped_GTEx_SV_eQTLs_all_associations.tsv | grep del | cut -f 1,2,6,7,10,11,12 | grep -e Brain_Hippocampus -e Liver -e Lung -e Muscle_Skeletal -e Pancreas -e  Heart_Left_Ventricle -e Artery_Aorta -e Small_Intestine_Terminal_Ileum > multiple_svs.tsv

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
from utils import GeneralUtils.parse_yaml
from utils import TISSUES_early_testing

negative_direction_eqtls = [
    [
        "hippocampus",
        "ENSG00000164744.12",
        "3_0_35239_del",
        -0.6438042031778736,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "hippocampus",
        "ENSG00000006377.10",
        "3_0_35239_del",
        -0.2936797367856676,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000078053.16",
        "3_0_35239_del",
        -0.2623219411917521,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000179869.14",
        "3_0_35239_del",
        -0.2615095075550178,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000135211.5",
        "3_0_35239_del",
        -0.25865248145267555,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "hippocampus",
        "ENSG00000170092.14",
        "3_0_35239_del",
        -0.2517748623987624,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000160870.12",
        "3_0_35239_del",
        -0.2502080745486415,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000158528.11",
        "3_0_35239_del",
        -0.24258394305078676,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000106336.12",
        "3_0_35239_del",
        -0.23531929357231152,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000127990.15",
        "3_0_35239_del",
        -0.2158726448663987,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000146834.13",
        "3_0_35239_del",
        -0.21461684646781262,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000128606.12",
        "3_0_35239_del",
        -0.2120187586062632,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000106608.16",
        "3_0_35239_del",
        -0.2067606451541328,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000155428.12",
        "3_0_35239_del",
        -0.20651630583744213,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "lung",
        "ENSG00000169894.17",
        "3_0_35239_del",
        -0.2034626845105262,
        "chr7",
        35046248,
        103160096,
    ],
]

positive_direction_eqtls = [
    [
        "hippocampus",
        "ENSG00000006634.7",
        "3_0_35239_del",
        0.20398663949133608,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000087085.13",
        "3_0_35239_del",
        0.20614364897025753,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "lung",
        "ENSG00000228204.2",
        "3_0_35239_del",
        0.207677492318611,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000130305.16",
        "3_0_35239_del",
        0.2107289132323575,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000146828.17",
        "3_0_35239_del",
        0.21470728664189329,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000106178.6",
        "3_0_35239_del",
        0.21515157976434116,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "lung",
        "ENSG00000128564.6",
        "3_0_35239_del",
        0.2168664546802869,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000001629.9",
        "3_0_35239_del",
        0.22185520521866678,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000187037.8",
        "3_0_35239_del",
        0.22483940128822572,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000197037.10",
        "3_0_35239_del",
        0.22826529142138344,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000201772.1",
        "3_0_35239_del",
        0.24060158825873143,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "hippocampus",
        "ENSG00000136206.3",
        "3_0_35239_del",
        0.2408891284701545,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "hippocampus",
        "ENSG00000126522.16",
        "3_0_35239_del",
        0.2421245581152866,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000185467.7",
        "3_0_35239_del",
        0.25980169193232105,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "hippocampus",
        "ENSG00000182487.12",
        "3_0_35239_del",
        0.261391431427422,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "liver",
        "ENSG00000146700.8",
        "3_0_35239_del",
        0.2641123645062674,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "lung",
        "ENSG00000226220.1",
        "3_0_35239_del",
        0.2785446881586681,
        "chr7",
        35046248,
        103160096,
    ],
    [
        "hippocampus",
        "ENSG00000106336.12",
        "3_0_35239_del",
        0.28444995173885645,
        "chr7",
        35046248,
        103160096,
    ],
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


def _eqtl_to_coords(eqtl, graph_idxs):
    """_summary_ of function"""
    tissue, gene, sv, beta, chrom, start, stop = eqtl
    idx = graph_idxs[f"{gene}_{tissue}"]
    begin = start - 1000000
    end = stop + 1000000
    return idx, start, stop, begin, end


def _get_subset_of_idx_dictionary(dictionary, chrom, tissue):
    """_summary_ of function"""
    return {
        key: value
        for key, value in dictionary.items()
        if chrom in key and tissue in key
    }


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
    savedir = "/ocean/projects/bio210019p/stevesho/data/preprocess/recapitulations/"

    # parse yaml for params, used to load data
    config = "/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs/ablation_experiments/curated.yaml"
    # parser = argparse.ArgumentParser()
    params = GeneralUtils.parse_yaml(config)

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
        graph_type="full",
        root_dir=root_dir,
        targets_types=params["training_targets"]["targets_types"],
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
    )

    def GeneralUtils._tensor_out_to_array(tensor, idx):
        return np.stack([x[idx].cpu().numpy() for x in tensor], axis=0)

    # set up loaders for inference
    batch_size = 2048
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
            savename = f"{qtl[0]}_{qtl[1]}_{qtl[2]}_{qtl[3]}"
            try:
                idx, start, stop, chr_window_start, chr_window_end = _eqtl_to_coords(
                    qtl, graph_idxs
                )
                chr_dict = _get_subset_of_idx_dictionary(
                    graph_idxs, chrom=qtl[4], tissue=qtl[0]
                )
                qtl_del_idxs = _get_idxs_for_sv(
                    chr_dict, graph_idxs, chr_window_start, chr_window_end
                )

                # store baseline
                qtl_dict[savename] = []
                qtl_dict[savename].append(baseline_expression[idx])

                data = graph_to_pytorch(
                    experiment_name=params["experiment_name"],
                    graph_type="full",
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
            except KeyError:
                pass

    with open(
        "/ocean/projects/bio210019p/stevesho/data/preprocess/pickles/qtl_dict.pkl", "wb"
    ) as file:
        pickle.dump(qtl_dict, file)

    # # try other tissues
    # other_tissues = {}

    # qtl = positive_direction_eqtls[1]
    # real_tissue = qtl[0]
    # for tissue in ['aorta', 'hippocampus', 'left_ventricle', 'pancreas', 'liver', 'lung', 'skeletal_muscle', 'small_intestine',]:
    #     if tissue != real_tissue:
    #         qtl[0] = tissue
    #         savename = f'{qtl[0]}_{qtl[1]}_{qtl[2]}_{qtl[3]}'
    #         try:
    #             idx, start, stop, chr_window_start, chr_window_end = _eqtl_to_coords(qtl, graph_idxs)
    #             chr_dict = _get_subset_of_idx_dictionary(graph_idxs, chrom=qtl[4], tissue=tissue)
    #             qtl_del_idxs = _get_idxs_for_sv(chr_dict, graph_idxs, chr_window_start, chr_window_end)

    #             # store baseline
    #             other_tissues[savename] = []
    #             other_tissues[savename].append(baseline_expression[idx])

    #             data = graph_to_pytorch(
    #                 experiment_name=params["experiment_name"],
    #                 graph_type='full',
    #                 root_dir=root_dir,
    #                 targets_types=params["training_targets"]["targets_types"],
    #                 test_chrs=params["training_targets"]["test_chrs"],
    #                 val_chrs=params["training_targets"]["val_chrs"],
    #                 node_remove_edges=[qtl_del_idxs],
    #             )
    #             perturb_loader = NeighborLoader(
    #                 data,
    #                 num_neighbors=[5, 5, 5, 5, 5, 3],
    #                 batch_size=batch_size,
    #                 input_nodes=data.all_mask,
    #             )
    #             _, _, _, perturbed_expression = all_inference(
    #                 model=model,
    #                 device=device,
    #                 data_loader=perturb_loader,
    #             )
    #             perturbed_expression = _average_values_across_dict(perturbed_expression)
    #             other_tissues[savename].append(perturbed_expression[idx])
    #         except KeyError:
    #             pass


if __name__ == "__main__":
    main()
