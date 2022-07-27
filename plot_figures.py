#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Code to plot various data"""

import pickle

import matplotlib.pyplot as plt
import seaborn as sns


def _cat_numnodes(files):
    num_nodes = []
    for file in files:
        with open(file, 'rb') as f:
            nodes = pickle.load(f)
        num_nodes = num_nodes + nodes
    num_nodes.sort()
    return num_nodes


if __name__ == '__main__':
    ### set matplotlib parameters
    plt.rcParams.update({'font.size': 7})  # set font size
    plt.rcParams["font.family"] = 'Helvetica'  # set font
    # plt.rcParams["figure.figsize"] = [34,18]

    ### get num_nodes across all graphs
    files = ['num_nodes_hippocampus.pkl', 'num_nodes_left_ventricle.pkl', 'num_nodes_mammary.pkl']
    all_nodes = _cat_numnodes(files)

    ### plot with smoothed function
    sns.distplot(all_nodes)
    plt.title('Distribution of num_nodes', fontsize=7)
    plt.xlabel('Number of graphs', fontsize=7)
    plt.ylabel('Number of nodes', fontsize=7)

    ### plot with absolute values
    sns.distplot(all_nodes, kde=False)
    plt.title('Distribution of num_nodes')
    plt.xlabel('Number of graphs')
    plt.ylabel('Number of nodes')