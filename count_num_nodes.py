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
def _num_nodes_from_graph(filename: str) -> int:
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    return graph['num_nodes']


def main() -> None:
    """Save num_nodes as array for each individual tissue"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tissue', type=str, required=True)
    args = parser.parse_args()

    gff = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed'
    genes = list(genes_from_gff(gff))

    directory = f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs'
    node_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/count_num_nodes'

    nodes = []
    for idx, gene in enumerate(genes):
        nodes.append(_num_nodes_from_graph(gene))

    with open(f'{node_dir}/num_nodes_{args.tissue}.pkl', 'wb') as output:
        pickle.dump(nodes, output)


if __name__ == '__main__':
    main()