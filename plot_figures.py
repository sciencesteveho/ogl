#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Code to plot various data"""

import csv
import pickle
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import pybedtools
import seaborn as sns

from collections import Counter


tissues = [
    'hippocampus',
    'left_ventricle',
    'mammary',
    'liver',
    'lung',
    'pancreas',
    'skeletal_muscle'
]

def _set_matplotlib_publication_parameters():
    plt.rcParams.update({'font.size': 7})  # set font size
    plt.rcParams["font.family"] = 'Helvetica'  # set font


def _cat_numnodes(files):
    num_nodes = []
    for file in files:
        with open(file, 'rb') as f:
            nodes = pickle.load(f)
        num_nodes = num_nodes + nodes
    num_nodes.sort()
    return num_nodes


def feat_list(file, col_idx):
    with open(file, newline='') as file:
        return [
            line[col_idx]
            for line in csv.reader(file, delimiter='\t')
        ]


def uniq_feats(feat_list):
    return Counter(feat_list)


def plot_num_nodes_dist(all_nodes, fontsize, g_type, kde_option=False):
    sns.displot(all_nodes, kde=kde_option)
    plt.title(f'Distribution, number of {g_type} per graph', fontsize=fontsize)
    plt.xlabel(f'Number of {g_type}', fontsize=fontsize)
    plt.ylabel('Number of graphs', fontsize=fontsize)
    return plt    


def plot_sparsity(zeros, fontsize):
    _set_matplotlib_publication_parameters()
    sns.displot(zeros)
    plt.title('Sparsity of adjacency matrices', fontsize=fontsize)
    plt.xlabel('Percentage of zeros in adj', fontsize=fontsize)
    plt.ylabel('Number of graphs', fontsize=fontsize)
    return plt    


def barplot_feats(uniq_feats):
    '''Allo'''


def merge_bedfiles(tissue):
    root_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/'
    cat_cmd = f"cat {root_dir}/{tissue}/parsing/attributes/gc/*_* | sort -k1,1 -k2,2n > {root_dir}/paring/all_nodes.txt"
    subprocess.run(cat_cmd, stdout=None, shell=True)
    nodes = pybedtools.BedTool(f"{root_dir}/parsing/all_nodes.txt")
    nodes = nodes.merge()

if __name__ == '__main__':
    ### set matplotlib parameters
    _set_matplotlib_publication_parameters()

    ### get num_nodes across all graphs
    # files = ['num_nodes_hippocampus.pkl', 'num_nodes_left_ventricle.pkl', 'num_nodes_mammary.pkl']
    # all_nodes = _cat_numnodes(files)

    with open('node_count.pkl', 'rb') as file:
        all_nodes = pickle.load(file)

    with open('edge_count.pkl', 'rb') as file:
        all_edges = pickle.load(file)

    ### plot with smoothed function
    smoothed_plt = plot_num_nodes_dist(all_nodes, 7, g_type='nodes', kde_option=True)  # nodes
    plt.savefig('node_count_v3.png', dpi=300, format='png', bbox_inches = "tight",)

    smoothed_plt = plot_num_nodes_dist(all_edges, 7, g_type='edges', kde_option=True)  # edges
    plt.savefig('edge_count_v3.png', dpi=300, format='png', bbox_inches = "tight",)

    ### plot with absolute values
    absolute_plt = plot_num_nodes_dist(all_nodes, 7, kde_option=False)

    ### uniq_feats for chromHMM
    feats = feat_list('E095_15_coreMarks_hg38lift_mnemonics.bed', 3)
    uniq = uniq_feats(feats)
    sns.barplot(x=list(uniq.keys()), y=list(uniq.values()), order=list(uniq.values()).sort())

    ### percentage zeroes
    with open('zero_percents.pkl', 'rb') as f:
        zeros = pickle.load(f)

    zeros = [float(zero.split('%')[0]) for zero in zeros]



#!/usr/bin/python

# import numpy as np
# import pybedtools

# import matplotlib.pyplot as plt
# import seaborn as sns


def _set_matplotlib_publication_parameters():
    plt.rcParams.update({'font.size': 7})  # set font size
    plt.rcParams["font.family"] = 'Helvetica'  # set font


def distance_plot(distances, num, fontsize=7):
    _set_matplotlib_publication_parameters()
    sns.displot(distances, kind='kde', color='maroon')
    plt.title(f'Distance between features, {num} kb', fontsize=fontsize)
    plt.xlabel('Distance', fontsize=fontsize)
    plt.ylabel('Counts', fontsize=fontsize)
    return plt    


def bed_distances(bed): 
    a = pybedtools.BedTool(bed)
    starts = np.array([int(line[1]) for line in a])
    ends = np.array([int(line[2]) for line in a])
    return [
        x for x in np.subtract(
        starts[1:len(starts)],
        ends[:len(ends)-1])
        if x > 0
    ]


_set_matplotlib_publication_parameters()
def plot_distances(arg):
    distances = bed_distances(arg[0])
    distance_plot(distances, arg[1])
    plt.savefig((f'{arg[1]}_kb_distance.png'), format='png', dpi=300, bbox_inches='tight')


kbs = [
    ('50000_liver.bed', '50'),
    ('75000_liver.bed', '75'),
    ('125000_liver.bed', '125'),
    ('250000_liver.bed', '250'),
     ]

for tup in kbs:
    print(tup)
    plot_distances(tup)



import csv
import pybedtools

def hg38_total(chrom):
    with open(chrom, newline='') as file:
        total = 0
        file_reader = csv.reader(file, delimiter = '\t')
        for idx, line in enumerate(file_reader):
            if idx <= 23:
                total += int(line[1])
    return total

def bed_total(bed):
    a = pybedtools.BedTool(bed)
    total = 0
    for line in a:
        total += len(line)
    return total

total_bp = hg38_total('hg38.chrom.sizes.txt')
for x in kbs:
    total = bed_total(x[0])
    print(f'{x}, {total/total_bp}')


'''
('50000_liver.bed', '50'), 0.5190694321428063
('75000_liver.bed', '75'), 0.5579010487189838
('125000_liver.bed', '125'), 0.6123175259512104
('250000_liver.bed', '250'), 0.6931871563870524
'''

