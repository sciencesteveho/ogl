
import csv
from ftplib import all_errors
import pickle
import subprocess
import networkx as nx

import numpy as np

from typing import Any, Dict, List, Tuple

# gene='ENSG00000128513.14'
# tissue='pancreas'
# interaction_file = f'/ocean/projects/bio210019p/stevesho/data/preprocess/{tissue}/interaction/interaction_edges.txt'
# parse_dir = f'/ocean/projects/bio210019p/stevesho/data/preprocess/{tissue}/parsing'
# int_nodes =  f'/ocean/projects/bio210019p/stevesho/data/preprocess/{tissue}/interaction/uniq_interaction_nodes.txt'

# with open(interaction_file, newline='') as file:
#     interaction_edges = [line for line in csv.reader(file, delimiter='\t')]

# print(f'starting _prepare_graph_tensors on {gene}')

# def _reindex_nodes(edges):
#     """_lorem"""
#     uniq_nodes = sorted(
#         set([edge[0] for edge in edges]+[edge[1] for edge in edges])
#         )
#     node_idxs = {node: id for id, node in enumerate(uniq_nodes)}
#     edges_reindexed = list(
#         map(lambda edge: [node_idxs[edge[0]], node_idxs[edge[1]], edge[2]], edges)
#         )
#     return sorted(edges_reindexed), node_idxs, len(uniq_nodes)

# gene_edges = f'{parse_dir}/edges/genes/{gene}'

# ### fast uniq_nodes 
# uniq_local_sort = f"awk '{{print $4 \"\\n\" $8}}' {gene_edges} \
#     | sort -u"
# proc = subprocess.Popen(uniq_local_sort, shell=True, stdout=subprocess.PIPE)
# uniq_local = proc.communicate()[0]

# with open(int_nodes) as f:
#     interaction_nodes = [line.rstrip('\n') for line in f.readlines()]

# nodes_to_add = set(str(uniq_local).split('\\n')).intersection(interaction_nodes)

# edges_to_add = [
#     [line[0], line[1], line[3]] for line in
#     filter(
#         lambda interaction: interaction[0] in nodes_to_add or interaction[1] in nodes_to_add,
#         interaction_edges
#     )
# ]

# with open(gene_edges, newline='') as file:
#     local_edges = [
#         [line[3], line[7], 'local']
#         for line in csv.reader(file, delimiter='\t')]


# edges = local_edges + edges_to_add
# edges_reindexed, node_idxs, num_nodes = _reindex_nodes(edges)


# with open(f'{tissue}_local_edges.txt', 'wb') as output:
#     pickle.dump(local_edges, output)

# with open(f'{tissue}_all_edges.txt', 'wb') as output:
#     pickle.dump(edges, output)

# with open(f'{tissue}_idxs.txt', 'wb') as output:
#     pickle.dump(edges, output)

import pickle
import networkx as nx
from pyvis.network import Network

gene='ENSG00000128513.14'
tissue='lung'

with open(f'{tissue}_local_edges.txt', 'rb') as f:
    local_edges = pickle.load(f)

with open(f'{tissue}_all_edges.txt', 'rb') as f:
    edges = pickle.load(f)

with open(f'{tissue}_idxs.txt', 'rb') as f:
    idxs = pickle.load(f)

# get uniq nodes
uniq_local = set([x[0] for x in local_edges] +[x[1] for x in local_edges])
uniq_all = set([x[0] for x in edges] +[x[1] for x in edges])

# initialize network
local = Network(height="1000px", width="100%")
all_e = Network(height="1000px", width="100%")

for x in uniq_local:
    local.add_node(x)

# for x in uniq_all:
#     all_e.add_node(x)

for x in local_edges:
    local.add_edge(x[0], x[1])

# for x in edges:
#     all_e.add_edge(x[0], x[1])

local.toggle_physics(True)

local.show(
    'local.html',
    )
local.show_buttons(filter_=['physics'])
# all_e.toggle_physics(True)

# all_e.show(
#     'all_e.html',
#     )