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


def main() -> None:
    """Save num_nodes as array for each individual tissue"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tissue', type=str, required=True)
    args = parser.parse_args()

    gff = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed'
    genes = list(genes_from_gff(gff))

    directory = f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs'
    node_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/count_num_nodes'

    graph_stats = {}
    for _, gene in enumerate(genes):
        graph_stats[gene] = _nodes_edges_from_graph(f'{directory}/{gene}_{args.tissue}')

    with open(f'{node_dir}/num_nodes_{args.tissue}.pkl', 'wb') as output:
        pickle.dump(graph_stats, output)


if __name__ == '__main__':
    main()