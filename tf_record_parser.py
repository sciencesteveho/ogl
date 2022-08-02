#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 

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
from multiprocessing import Pool

# from utils import dir_check_make

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
        default="tfrecords_min",
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
        default="genomic-graph-mutagenesis",
        help="name of the dataset to call GGraphMutagenesisTFRecordProcessor with",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="genomic-graph-mutagenesis",
        help="name of the dataset; i.e. prefix to use for TFRecord names",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default="targets_filtered_random_1000.pkl",
    )

    args = parser.parse_args(sys.argv[1:])
    params = get_params(args.params)

    # Only update params that are not None
    cmd_params = list(vars(args).items())
    params.update({k: v for (k, v) in cmd_params if v is not None})

    return params


class GGraphMutagenesisTFRecordProcessor:
    """Data Processor for handling OGB's graph prediction data.

    :param dict params: Model configuration parameters.
    :param int feature_dim: Size of feature dimension for convolution.
        Defaults to 9.
    :param bool normalize: Specifies whether to normalize the labels for
        training. Defaults to False.
    """

    def __init__(
        self, params, name, output_dir, output_name, num_files, target_file, feature_dim=9
    ):
        self.name = name
        self.feature_dim = feature_dim
        self.shuffle_raw_train_data = params.get("shuffle_raw_train_data", True)
        self.max_num_nodes = params.get("max_num_nodes", 9999)
        self.task_type = params.get("task_type", "binary_classification")
        self.normalize = params.get("normalize", True)
        self.mode = params['mode']

        self.output_dir = output_dir 
        self.output_name = output_name
        self.num_files = max(num_files, 1)
        self.target_file = target_file
        ### add targets
        path = 'shared_data'
        with open(f'{path}/{self.target_file}', 'rb') as f:
            targets = pickle.load(f)
        self.targets = targets

    def _prepare_output_processor(self, idx):
        tfrecords_dir = self.output_dir + f'_{idx}'
        # dir_check_make(tfrecords_dir)

        output_files = [
            os.path.join(
                tfrecords_dir, '%s_%i.tfrecords' % (self.output_name, fidx + 1)
            )
            for fidx in range(self.num_files)
        ]

        return output_files

    def _open_graph(self, gene):
        '''utility to open the graph file'''
        tissue = gene.split('_')[1:]
        with open(f"{'_'.join(tissue)}/parsing/graphs/{gene}", 'rb') as f:
            return pickle.load(f)

    def create_tfrecords(self, split_idx, output_idx):
        """
        Create TFRecrods from graphs, labels

        :param bool is_training: Specifies whether the data is for training
        :returns: tf dataset
        """

        writers = []
        for output_file in self._prepare_output_processor(output_idx):
            writers.append(tf.io.TFRecordWriter(output_file))
        writer_index = 0
        total_written = 0

        if self.mode == "train" and self.shuffle_raw_train_data:
            random.shuffle(split_idx)

        print("Processing to TFRecords ...")
        for idx in tqdm(split_idx, total=len(split_idx)):
            graph = self._open_graph(idx)
            label = self.targets[self.mode][idx]
            num_nodes = graph["num_nodes"]
            node_feat = graph["node_feat"].numpy()

            # form adj. matrix
            row, col = graph["edge_index"].numpy()
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
            features["node_feat"] = self._create_float_feature(node_feat)
            features["node_mask"] = self._create_float_feature(node_mask)
            features["label"] = self._create_float_feature(label.astype(np.int64))

            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features)
            )

            writers[writer_index].write(tf_example.SerializeToString())
            writer_index = (writer_index + 1) % len(writers)

            total_written += 1

        for writer in writers:
            writer.close()

    def _create_int_feature(self, values):
        """Returns an int64_list from a bool / enum / int / uint."""
        if values is None:
            values = []
        if isinstance(values, np.ndarray) and values.ndim > 1:
            values = values.reshape(-1)
        if not isinstance(values, list):
            values = values.tolist()
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def _create_float_feature(self, values):
        """Returns a float_list from a float / double."""
        if values is None:
            values = []
        if isinstance(values, np.ndarray) and values.ndim > 1:
            values = values.reshape(-1)
        if not isinstance(values, list):
            values = values.tolist()
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))


    def _normalize_adj(self, A, symmetric=False):
        """
        Computes the graph filter described in
        [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).

        :param A: array with rank 2;
        :param symmetric: boolean, whether to normalize the matrix as
        \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);
        :return: array with rank 2, same as A;
        """
        fltr = A.copy()
        I = np.eye(A.shape[-1], dtype=A.dtype)
        A_tilde = A + I
        fltr = self._degrees(A_tilde, symmetric=symmetric)
        return fltr

    def _degrees(self, A, symmetric):
        """
        Normalizes the given adjacency matrix using the degree matrix as either
        \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).

        :param A: rank 2 array
        :param symmetric: boolean, compute symmetric normalization;
        :return: the normalized adjacency matrix.
        """
        if symmetric:
            normalized_D = self._degree_power(A, -0.5)
            output = normalized_D.dot(A).dot(normalized_D)
        else:
            normalized_D = self._degree_power(A, -1.0)
            output = normalized_D.dot(A)
        return output

    def _degree_power(self, A, k):
        """
        Computes \(\D^{k}\) from the given adjacency matrix.

        :param A: rank 2 array or sparse matrix.
        :param k: exponent to which elevate the degree matrix.
        :return: D^k
        """
        degrees = np.power(np.array(A.sum(1)), k).flatten()
        degrees[np.isinf(degrees)] = 0.0
        return np.diag(degrees)


    def _get_split_idx(self):
        return {"train": list(self.targets['train'].keys()), 
        "validation": list(self.targets['validation'].keys()), 
        "test": list(self.targets['test'].keys())
        }


if __name__ == "__main__":
    params = get_arguments()

    name = params["name"]
    output_name = params["output_name"]
    output_dir = params["output_dir"]
    num_files = params["num_files"]
    target_file = params['target_file']
    mode = params["mode"]
    if mode == "train":
        init_params = params["train"]
    elif mode == "validation":
        init_params = params["validation"]
    elif mode == "test":
        init_params = params["test"]

    ### initialize the object
    ogbObject = GGraphMutagenesisTFRecordProcessor(
        init_params, name, output_dir+'/'+mode, output_name, num_files, target_file
    )
    
    ### process data in parallel
    cores=12
    split_list = ogbObject._get_split_idx()[mode]
    split_idxs = list(range(0,60,5))
    split_pool = np.array_split(split_list, 24)
    pool = Pool(processes=12)
    pool.starmap(ogbObject.create_tfrecords, zip(split_pool, split_idxs))
    pool.close()

    print("Data preprocessing complete.")