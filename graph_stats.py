#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 

"""Store num_nodes in array"""

import argparse
import pickle

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
    return sum([graph_stats[gene][0] for gene in graph_stats.keys()]), sum([graph_stats[gene][1] for gene in graph_stats.keys()])


def main() -> None:
    """Save num_nodes as array for each individual tissue"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tissue', type=str, required=True)
    args = parser.parse_args()

    graph_stats = _edge_and_nodes_per_gene(
        gff = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed',
        directory = f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs',
        tissue=args.tissue
        )

    ### save
    node_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/count_num_nodes'
    with open(f'{node_dir}/num_nodes_{args.tissue}.pkl', 'wb') as output:
        pickle.dump(graph_stats, output)

    tissue_params = [
        ('num_nodes_hippocampus.pkl', 'hippocampus'),
        ('num_nodes_left_ventricle.pkl', 'left_ventricle'),
        ('num_nodes_mammary.pkl', 'mammary'),   
        ]

    ### unfiltered
    all_stats_dict = _cat_stat_dicts(tissue_params)
    total_nodes, total_edges = _summed_counts(all_stats_dict)

    with open('targets_filtered.pkl', 'rb') as f:
        filtered_targets = pickle.load(f)

    filtered_genes = list(filtered_targets['train'].keys()) + list(filtered_targets['test'].keys()) + list(filtered_targets['validation'].keys())

    filtered_stats = {
        gene: value for gene, value
        in all_stats_dict.items()
        if gene in filtered_genes
    }
    total_nodes, total_edges = _summed_counts(filtered_stats)
    
if __name__ == '__main__':
    main()