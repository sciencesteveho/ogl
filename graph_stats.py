#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 

"""Store num_nodes in array"""

import argparse
import pickle
import numpy as np
import scipy.sparse as sp

from utils import genes_from_gff, time_decorator


@time_decorator(print_args=True)
def _nodes_edges_from_graph(filename: str) -> int:
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    return graph['num_nodes'], graph['edge_index'].shape[1]


def _open_and_add_tissue(filename, tissue):
    with open(filename, 'rb') as file:
        stats = pickle.load(file)

    return {
        f'{gene}_{tissue}':value for gene, value in stats.items()
    }


def _cat_stat_dicts(tissue_params):
    for idx, tup in enumerate(tissue_params):
        if idx == 0:
            shared = _open_and_add_tissue(tup[0], tup[1])
        else:
            update = _open_and_add_tissue(tup[0], tup[1])
            shared.update(update)
    return shared


def _edge_and_nodes_per_gene(
    gff,
    directory,
    tissue,
    ):
    genes = list(genes_from_gff(gff))
    return {
        gene:_nodes_edges_from_graph(f'{directory}/{gene}_{tissue}')
        for _, gene in enumerate(genes)
    }


def _summed_counts(graph_stats):
    return (
        sum([graph_stats[gene][0] for gene in graph_stats.keys()]),
        sum([graph_stats[gene][1] for gene in graph_stats.keys()])
    )


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
    return (adj.size, nonzero, "{:.1%}".format(sparse_percentage))


def main() -> None:
    """Save num_nodes as array for each individual tissue"""
    # ctcf = 'ENSG00000102974.14'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--tissue', type=str, required=True)
    # args = parser.parse_args()

    # graph_stats = _edge_and_nodes_per_gene(
    #     gff = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed',
    #     directory = f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs',
    #     tissue=args.tissue
    #     )

    # ### save
    # node_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/count_num_nodes'
    # with open(f'{node_dir}/num_nodes_{args.tissue}.pkl', 'wb') as output:
    #     pickle.dump(graph_stats, output)

    # tissue_params = [
    #     ('num_nodes_hippocampus.pkl', 'hippocampus'),
    #     ('num_nodes_left_ventricle.pkl', 'left_ventricle'),
    #     ('num_nodes_mammary.pkl', 'mammary'),   
    #     ]

    # ### unfiltered
    # all_stats_dict = _cat_stat_dicts(tissue_params)
    # total_nodes, total_edges = _summed_counts(all_stats_dict)

    # with open('targets_filtered.pkl', 'rb') as f:
    #     filtered_targets = pickle.load(f)

    # filtered_genes = list(filtered_targets['train'].keys()) + list(filtered_targets['test'].keys()) + list(filtered_targets['validation'].keys())

    # filtered_stats = {
    #     gene: value for gene, value
    #     in all_stats_dict.items()
    #     if gene in filtered_genes
    # }
    # total_nodes, total_edges = _summed_counts(filtered_stats)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tissue', type=str, required=True)
    # parser.add_argument('-p', '--partition', type=str, required=True)
    args = parser.parse_args()

    savedir = '/ocean/projects/bio210019p/stevesho/data/preprocess/sparse_check'
    directory = f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs'
    gff = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed'

    genes = list(genes_from_gff(gff))
    sparse = []
    for gene in genes:
        sparse.append(_graph_sparsity(f'{directory}/{gene}_{args.tissue}'))
        print(f'{gene}')

    with open(f'{savedir}/sparse_vals_{args.tissue}.pkl', 'wb') as output:
        pickle.dump(sparse, output)
    
    
if __name__ == '__main__':
    main()