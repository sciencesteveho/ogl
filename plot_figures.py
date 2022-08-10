#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Code to plot various data"""

import csv
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter


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


def plot_num_nodes_dist(all_nodes, fontsize, kde_option=False):
    sns.distplot(all_nodes, kde=kde_option)
    plt.title('Distribution of num_nodes', fontsize=fontsize)
    plt.xlabel('Number of graphs', fontsize=fontsize)
    plt.ylabel('Number of nodes', fontsize=fontsize)
    return plt


def barplot_feats(uniq_feats):
    '''Allo'''


if __name__ == '__main__':
    ### set matplotlib parameters
    _set_matplotlib_publication_parameters()

    ### get num_nodes across all graphs
    files = ['num_nodes_hippocampus.pkl', 'num_nodes_left_ventricle.pkl', 'num_nodes_mammary.pkl']
    all_nodes = _cat_numnodes(files)

    ### plot with smoothed function
    smoothed_plt = plot_num_nodes_dist(all_nodes, 7, kde_option=True)

    ### plot with absolute values
    absolute_plt = plot_num_nodes_dist(all_nodes, 7, kde_option=False)

    ### uniq_feats for chromHMM
    feats = feat_list('chromhmm_mammary.bed', 3)
    uniq = uniq_feats(feats)
    sns.barplot(x=list(uniq.keys()), y=list(uniq.values()), order=list(uniq.values()).sort())



idx=0
for i in all_nodes:
    if i <= 7500:
        idx+=1