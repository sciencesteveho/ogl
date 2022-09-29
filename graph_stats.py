#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 

"""Store num_nodes in array"""

import argparse
import pickle
import os

import numpy as np
import scipy.sparse as sp

import pybedtools

from utils import genes_from_gff, filtered_genes, time_decorator


@time_decorator(print_args=True)
def _nodes_edges_from_graph(filename: str) -> int:
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    return graph['num_nodes'], graph['edge_index'].shape[1]


def _cat_stat_dicts(tissue_params):
    """"""
    def _open_and_add_tissue(filename, tissue):
        with open(filename, 'rb') as file:
            stats = pickle.load(file)

        return {
            f'{gene}_{tissue}':value for gene, value in stats.items()
        }
    for idx, tup in enumerate(tissue_params):
        if idx == 0:
            shared = _open_and_add_tissue(tup[0], tup[1])
        else:
            update = _open_and_add_tissue(tup[0], tup[1])
            shared.update(update)
    return shared


def _summed_counts(graph_stats):
    return (
        sum([graph_stats[gene][0] for gene in graph_stats.keys()]),
        sum([graph_stats[gene][1] for gene in graph_stats.keys()])
    )


def _edge_and_nodes_per_gene(
    genes,
    directory,
    tissue,
    ):
    return {
        gene:_nodes_edges_from_graph(f'{directory}/{gene}_{tissue}')
        for _, gene in enumerate(genes)
    }


@time_decorator(print_args=True)
def _graph_sparsity(filename):
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    num_nodes = graph["num_nodes"]
    row, col = graph["edge_index"].numpy()
    adj = sp.coo_matrix(
        (np.ones_like(row),
        (row, col)),
        shape=(num_nodes, num_nodes)
        ).toarray()
    zeros = np.count_nonzero(adj)
    nonzero = adj.size - zeros
    sparse_percentage = zeros/adj.size
    return (adj.size, nonzero, "{:.1%}".format(sparse_percentage))  # tup[2] is calculated incorrectly


def _percentage_of_zeroes(tup_list):
    tups = []
    for lst in tup_list:
        with open(lst, "rb") as file:
            intermediate_list = pickle.load(file)
        tups += intermediate_list
    return[
        "{:.2%}".format(tup[1]/tup[0])
        for tup in tups
    ]


def main() -> None:
    # """Save num_nodes as array for each individual tissue"""
    # ctcf = 'ENSG00000102974.14'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--tissue', type=str, required=True)
    # args = parser.parse_args()

    # root_dir='/ocean/projects/bio210019p/stevesho/data/preprocess'
    # node_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/check_num_nodes'

    # genes = filtered_genes(f'{root_dir}/{args.tissue}/gene_regions_tpm_filtered.bed')

    # graph_stats = _edge_and_nodes_per_gene(
    #     genes=genes,
    #     directory=f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs',
    #     tissue=args.tissue
    #     )

    # ### save
    # with open(f'{node_dir}/num_nodes_{args.tissue}.pkl', 'wb') as output:
    #     pickle.dump(graph_stats, output)


    node_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/check_num_nodes'
    tissue_params = [
        ['num_nodes_hippocampus.pkl', 'hippocampus'],
        ['num_nodes_left_ventricle.pkl', 'left_ventricle'],
        ['num_nodes_mammary.pkl', 'mammary'],
        ['num_nodes_liver.pkl', 'liver'],
        ['num_nodes_lung.pkl', 'lung'],
        ['num_nodes_pancreas.pkl', 'pancreas'],
        ['num_nodes_skeletal_muscle.pkl', 'skeletal_muscle'],
        ]

    tissue_params = [[node_dir + '/' + x[0], x[1]] for x in tissue_params]

    ### unfiltered
    all_stats_dict = _cat_stat_dicts(tissue_params)
    # total_nodes, total_edges = _summed_counts(all_stats_dict)

    node_counts = [x[0] for x in all_stats_dict.values()]
    edge_counts = [x[1] for x in all_stats_dict.values()]

    ### save 
    with open(f'{node_dir}/node_count.pkl', 'wb') as output:
        pickle.dump(node_counts, output)

    with open(f'{node_dir}/edge_count.pkl', 'wb') as output:
        pickle.dump(edge_counts, output)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--tissue', type=str, required=True)
    # # parser.add_argument('-p', '--partition', type=str, required=True)
    # args = parser.parse_args()

    # savedir = '/ocean/projects/bio210019p/stevesho/data/preprocess/sparse_check'
    # directory = f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs'
    # gff = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed'

    # genes = list(genes_from_gff(gff))
    # sparse = []
    # for gene in genes:
    #     sparse.append(_graph_sparsity(f'{directory}/{gene}_{args.tissue}'))

    # with open(f'{savedir}/sparse_vals_{args.tissue}.pkl', 'wb') as output:
    #     pickle.dump(sparse, output)

    # sparse_files = ['sparse_vals_hippocampus.pkl', 'sparse_vals_left_ventricle.pkl', 'sparse_vals_mammary.pkl']
    # percent_zeros_in_graphs = _percentage_of_zeroes(sparse_files)
    
    
if __name__ == '__main__':
    main()