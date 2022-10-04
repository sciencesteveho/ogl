# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path as osp

import tensorflow as tf

from modelzoo.common.tf.model_utils.shard_dataset import shard_dataset


class OGBGMOLCHEMBLDataProcessor:
    """
    Data Processor for handling OGB's graph prediction data.

    :param dict params: Model configuration parameters.
    :param int feature_dim: Number of node features.
        Defaults to 9 (for OGBG-MOL* data sets)
    """

    def __init__(self, params, feature_dim=34):
        self.batch_size = params["batch_size"]
        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", None)
        self.repeat = params.get("repeat", True)
        self.use_multiple_workers = params.get("use_multiple_workers", False)
        self.n_parallel_reads = params.get("n_parallel_reads", 4)

        assert self.batch_size > 0, "Batch size should be positive."

        self.root_dir = params["dataset_dir"]

        # set mixed precision parameters
        self.mixed_precision = params.get("mixed_precision", True)
        self.mp_type = tf.float16 if self.mixed_precision else tf.float32

        self.feature_dim = feature_dim
        self.max_num_nodes = params.get("max_num_nodes", 4000)
        self.num_targets = params.get("num_targets", 4)
        # for sharding on the Cerebras System, we need to explicitly retrieve TF_CONFIG
        self.label_mask_value = -1
        self.num_inputs = len(params["atom_feats"])

    def post_batch_map_fn(self, features, labels):
        mask = tf.not_equal(labels, self.label_mask_value)
        mask = tf.cast(mask, self.mp_type)

        # calculate the number of valid instances in batch
        num_valid = tf.reduce_sum(mask)
        num_valid = tf.cond(
            tf.equal(num_valid, tf.cast(0, self.mp_type)),
            lambda: tf.cast(1, self.mp_type),
            lambda: num_valid,
        )
        # adjusting to account for tf.reduce_mean
        num_valid = tf.math.divide(self.batch_size, num_valid)
        _size, _num_targets = tf.shape(labels)[0], tf.shape(labels)[1]
        num_valid = tf.broadcast_to(num_valid, [_size,])

        features["label_num_valid"] = num_valid
        return (features, labels)

    def map_fn(self, raw_record):
        """Parses a serialized protobuf example into a dictionary
        of input features and labels for Graph training.
        """
        feature_map = {
            "adj": tf.io.FixedLenFeature(
                [self.max_num_nodes, self.max_num_nodes], tf.float32
            ),
            "node_feat": tf.io.FixedLenFeature(
                [self.max_num_nodes, self.feature_dim], tf.float32
            ),
            "node_mask": tf.io.FixedLenFeature(
                [self.max_num_nodes, 1], tf.float32
            ),
            "label": tf.io.FixedLenFeature([self.num_targets,], tf.float32),
        }

        example = tf.io.parse_single_example(raw_record, feature_map)
        for name in list(example.keys()):
            feature = example[name]
            if feature.dtype == tf.float32:
                feature = tf.cast(feature, tf.float32)
                example[name] = feature

        feature = {
            "adj": tf.cast(example["adj"], self.mp_type),
            "node_mask": tf.cast(example["node_mask"], tf.float32),
        }

        node_feat = example["node_feat"]
        for i in range(self.num_inputs):
            name = "node_feat" + str(i)
            feature[name] = tf.cast(node_feat[..., i], tf.float32)
        label = tf.cast(example["label"], self.mp_type)

        return (feature, label)

    def create_tf_dataset(
        self, mode=tf.estimator.ModeKeys.TRAIN, input_context=None,
    ):
        """
        Create tf dataset.

        :param tf.estimator.ModeKeys mode: Specifies execution mode: train, eval, predict
        :param tf.distribute.InputContext input_context: Contains information
        about compute replicas and input pipelines in distributed mode
        :returns: tf dataset
        """

        is_training = mode == tf.estimator.ModeKeys.TRAIN
        if mode == tf.estimator.ModeKeys.TRAIN:
            split_name = "train"
        elif mode == tf.estimator.ModeKeys.EVAL:
            split_name = "valid"
        elif mode == tf.estimator.ModeKeys.PREDICT:
            split_name = "test"
        else:
            raise ValueError(
                f"mode should be one of train, eval or predict, got {mode}"
            )
        _drop_remainder = not (mode == tf.estimator.ModeKeys.PREDICT)
        file_pattern = osp.join(
            self.root_dir, split_name, "*.tfrecords*"
        )
        filelist = tf.data.Dataset.list_files(
            file_pattern, shuffle=self.shuffle, seed=self.shuffle_seed
        )
        filelist = shard_dataset(
            filelist, self.use_multiple_workers, input_context
        )

        dataset = filelist.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=self.n_parallel_reads,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            # only allow nondeterminism when shuffling unseeded
            deterministic=not (self.shuffle and self.shuffle_seed is None),
        )

        return self.transform_dataset(
            dataset,
            self.map_fn,
            self.batch_size,
            is_training,
            shuffle=self.shuffle,
            shuffle_buffer=self.shuffle_buffer,
            repeat=self.repeat,
            seed=self.shuffle_seed,
            post_batch_map_fn=self.post_batch_map_fn,
        )

    def transform_dataset(
        self,
        dataset,
        map_fn,
        batch_size,
        is_training,
        shuffle,
        shuffle_buffer=None,
        repeat=True,
        seed=None,
        post_batch_map_fn=None,
    ):
        """
        Apply standard transformations to a dataset:
            - shuffle -> map -> batch -> repeat if post_batch_map is False
            - shuffle -> map -> batch -> post_batch_map -> repeat
                if post_batch_map is True
        Batching before mapping is generally faster and the preferred method due to
        vectorization of map fn.
        Note: Mapping before batching may be required if parsing TF records that
        contain `FixedLenSequenceFeature` examples (rather than `FixedLenFeature`)
        :param tf.data.Dataset dataset: Dataset to apply transformations to
        :param func map_fn: Mapping function to be applied after batching data
        :param int batch_size: Batch size for model training
        :param bool shuffle: If True, then shuffle the dataset
        :param int shuffle_buffer: Size of shuffle buffer to sample data from
        :param bool repeat: If True, repeat the dataset
        :param int seed: Seed to use for shuffle randomizer or None
        :returns: tf dataset
        """

        if is_training and shuffle:
            if not shuffle_buffer:
                shuffle_buffer = 10 * batch_size
            dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)

        if map_fn:
            dataset = dataset.map(
                map_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                # only allow nondeterminism when shuffling unseeded
                deterministic=not (shuffle and seed is None),
            )

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        if post_batch_map_fn:
            dataset = dataset.map(
                post_batch_map_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                # only allow nondeterminism when shuffling unseeded
                deterministic=not (shuffle and seed is None),
            )

        if is_training and repeat:
            dataset = dataset.repeat()

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)