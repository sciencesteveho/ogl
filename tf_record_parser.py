#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Look into synder paper for mass spec values as a potential target

"""Convert tensors to TFrecords for running on CS-2

Much of this code is adapted from:
    cerebras/modelzeoo/graphs/tf/input/preprocess.py
"""


import argparse
import collections
import os
import os.path as osp
import pickle
import random
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import yaml
from tqdm import tqdm


def get_params(params_file):
    # Load yaml into params
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--params",
        type=str,
        default="data.yaml",
        help="Path to .yaml file with data parameters",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Path to .yaml file with data parameters",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tfrecords",
        help="directory name where TFRecords will be saved",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=16,
        help="number of files on disk to separate records into",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ogbg-molchembl",
        help="name of the dataset to call OGBGDataProcessor with",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="ogbg-molchembl",
        help="name of the dataset; i.e. prefix to use for TFRecord names",
    )

    args = parser.parse_args(sys.argv[1:])
    params = get_params(args.params)

    # Only update params that are not None
    cmd_params = list(vars(args).items())
    params.update({k: v for (k, v) in cmd_params if v is not None})

    return params


class OGBGTFRecordProcessor:
    """Data Processor for handling OGB's graph prediction data.

    :param dict params: Model configuration parameters.
    :param int feature_dim: Size of feature dimension for convolution.
        Defaults to 9.
    :param bool normalize: Specifies whether to normalize the labels for
        training. Defaults to False.
    """

    def __init__(
        self, params, name, output_dir, output_name, num_files, feature_dim=9
    ):
        self.name = name
        self.feature_dim = feature_dim
        self.shuffle_raw_train_data = params.get("shuffle_raw_train_data", True)
        self.split = params.get("split", "scaffold")
        self.max_num_nodes = params.get("max_num_nodes", 41200)
        self.task_type = params.get("task_type", "binary_classification")
        self.normalize = params.get("normalize", True)

        root_dir = params["root_dir"]
        self.dir_name = "_".join(self.name.split("-"))
        self.original_root_dir = root_dir
        self.root_dir = osp.join(root_dir, self.dir_name)

        self.output_dir = output_dir
        self.output_name = output_name
        self.num_files = max(num_files, 1)

        # initialize file reading once
        self._prepare_output_processor()

    def _prepare_output_processor(self):
        processed_dir = osp.join(self.root_dir, "processed")
        tfrecords_dir = osp.join(processed_dir, self.output_dir)
        if not osp.isdir(tfrecords_dir):
            os.makedirs(tfrecords_dir)

        output_files = [
            os.path.join(
                tfrecords_dir, '%s_%i.tfrecords' % (self.output_name, fidx + 1)
            )
            for fidx in range(self.num_files)
        ]

        self.output_files = output_files
        self.pre_processed_file = osp.join(
            processed_dir, "processed_graphs.pickle"
        )

    def _open_graph(self):
        '''utility to open the graph file'''


    def create_tfrecords(self, mode="train"):
        """
        Create TFRecrods from graphs, labels

        :param bool is_training: Specifies whether the data is for training
        :returns: tf dataset
        """

        writers = []
        for output_file in self.output_files:
            writers.append(tf.io.TFRecordWriter(output_file))
        writer_index = 0
        total_written = 0

        split_idx = self._get_split_idx(self.split)[mode]
        if mode == "train" and self.shuffle_raw_train_data:
            random.shuffle(split_idx)

        print("Processing to TFRecords ...")
        for idx in tqdm(split_idx, total=len(split_idx)):
            graph = graphs[idx]
            label = labels[idx]
            num_nodes = graph["num_nodes"]
            edge_feat = graph["edge_feat"]
            edge_index = graph["edge_index"]

            # get node features from edge features if none exists
            node_feat = (
                self._compute_node_feat(edge_feat, edge_index, reduce="sum")
                if graph["node_feat"] is None
                else graph["node_feat"]
            )
            # form adj. matrix
            row, col = graph["edge_index"]
            adj = sp.coo_matrix(
                (np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes)
            ).toarray()
            if self.normalize:
                adj = self._normalize_adj(adj)

            # get associated node_mask
            node_mask = np.ones((num_nodes, 1), dtype=np.float32)

            pad_num_nodes = self.max_num_nodes - num_nodes
            pad_value = 0

            adj = np.pad(
                adj,
                (0, pad_num_nodes),
                'constant',
                constant_values=(pad_value, pad_value),
            )
            node_feat = np.pad(
                node_feat,
                ((0, pad_num_nodes), (0, 0)),
                'constant',
                constant_values=(pad_value, pad_value),
            )
            node_mask = np.pad(
                node_mask,
                ((0, pad_num_nodes), (0, 0)),
                'constant',
                constant_values=(pad_value, pad_value),
            )

            features = collections.OrderedDict()
            features["adj"] = self._create_float_feature(adj.astype(np.float32))
            features["node_feat"] = self._create_int_feature(node_feat)
            features["node_mask"] = self._create_float_feature(node_mask)
            features["label"] = self._create_int_feature(label.astype(np.int64))

            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features)
            )

            writers[writer_index].write(tf_example.SerializeToString())
            writer_index = (writer_index + 1) % len(writers)

            total_written += 1

        for writer in writers:
            writer.close()



    
def main() -> None:
    """Pipeline to generate dataset split and target values"""

if __name__ == '__main__':
    main()


'''
self.num_files = 400 if each is .375

'''