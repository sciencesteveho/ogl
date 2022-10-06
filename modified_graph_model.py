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

import tensorflow as tf
from tensorflow import keras #

from modelzoo.common.tf.layers.DenseLayer import DenseLayer
# from modelzoo.common.tf.layers.EmbeddingLayer import EmbeddingLayer

from modelzoo.common.tf.layers.GraphAttentionLayer import GraphAttentionLayer
from modelzoo.common.tf.layers.GraphConvolutionLayer import (
    GraphConvolutionLayer,
)

from modelzoo.common.tf.layers.SquaredErrorLayer import SquaredErrorLayer
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel

__all__ = [
    "GCNModel",
    "GATModel",
]


class GNN(TFBaseModel):
    """
    Graph Neural Networks Base class. Inherits from TFBaseModel
    Provides functionality to build these types of graph networks:
    (1) Graph Convolutional Network
    (2) Graph Attention Network
    To use (1), pass model.model in params file as `GCNModel`
    To use (2), pass model.model in params file as `GATModel`
    :param dict params: Model configuration parameters.
    """

    def __init__(self, params):
        super(GNN, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )
        self.hidden_dim = params["model"]["hidden_dim"]
        self.gnn_depth = params["model"]["gnn_depth"]
        self.fc_depth = params["model"]["fc_depth"]
        self.activation = params["model"]["activation"]

        # attention, dropout, layer norm parameters
        self.attention = params["model"]["attention"]
        self.num_heads = params["model"]["num_heads"]
        # attention reduction mode for GAT layers
        self.attn_reduction = params["model"]["attention_reduction"]
        self.concat_heads = self.attn_reduction != "mean_reduction"
        self.dropout_rate = params["model"]["dropout_rate"]
        self.use_bias = params["model"]["use_bias"]
        self.layer_norm_epsilon = params["model"]["layer_norm_epsilon"]

        self.node_feats = params["train_input"]["atom_feats"]
        self.edge_feats = params["train_input"]["bond_feats"]
        self.num_targets = params["train_input"]["num_targets"]
        self.batch_size = params["train_input"]["batch_size"]

        assert (
            self.dropout_rate >= 0 and self.dropout_rate <= 1
        ), "Dropout rate should be in the range [0,1]"

        # optimizer and mixed precision settings
        self.tf_summary = params["model"]["tf_summary"]
        self.boundary_casting = params["model"]["boundary_casting"]
        self.mixed_precision = params["model"]["mixed_precision"]
        self.mp_type = tf.float16 if self.mixed_precision else tf.float32

        # Model trainer
        self.trainer = Trainer(
            params=params["optimizer"],
            tf_summary=self.tf_summary,
            mixed_precision=self.mixed_precision,
        )

        # adding alternative loss
        self.squared_error = SquaredErrorLayer()  

    def build_model(self, features, mode):
        """
        Build model and return output
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT,
        ], f"A correct estimator ModeKey is not passed."
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        relu_layer, fc_layers = self._build_functional_layers()
        graph_layers = self._build_graph_layers()
        adj = features["adj"]
        node_mask = features["node_mask"]

        # replace embedding with relu projection
        # node_mask = tf.cast(node_mask, self.mp_type)
        output = tf.multiply(
            node_mask, relu_layer(features["node_feat"])
            )

        # output = tf.math.accumulate_n(
        #     [
        #         tf.multiply(
        #             node_mask, emb_layer(features["node_feat" + str(i)])
        #         )
        #         for i, emb_layer in enumerate(emb_layers)
        #     ]
        # )
        ### adjust types back
        # node_mask = tf.cast(node_mask, self.mp_type)
        # output = tf.cast(output, self.mp_type)

        output = self._call_graph_layers(graph_layers, output, adj, is_training)

        output = tf.multiply(node_mask, output)
        output = tf.reduce_mean(output, axis=1)

        for layer in fc_layers:
            output = layer(output)

        return output

    def _call_graph_layers(self, graph_layers, output, adj, is_training):
        raise NotImplementedError("To be implemented child class!!")

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.trainer.build_train_ops(total_loss)

    def _build_functional_layers(self):
        """
        Build graph layers for the model
        """
        # build atom embedding
        # emb_layers = [
        #     EmbeddingLayer(
        #         input_dim=node_feat,
        #         output_dim=self.hidden_dim,
        #         boundary_casting=self.boundary_casting,
        #         tf_summary=self.tf_summary,
        #         dtype=self.policy,
        #     )
        #     for node_feat in self.node_feats
        # ]

        # dense relu
        relu_layer = DenseLayer(
                units=self.hidden_dim,
                activation=self.activation,
                boundary_casting=self.boundary_casting,
                tf_summary=self.tf_summary,
                dtype=self.policy,
            )

        # build fc layers for the model
        fc_layers = []
        for _ in range(self.fc_depth - 1):
            fc_layers.append(
                DenseLayer(
                    units=self.hidden_dim,
                    activation=self.activation,
                    boundary_casting=self.boundary_casting,
                    tf_summary=self.tf_summary,
                    dtype=self.policy,
                )
            )
        fc_layers.append(
            DenseLayer(
                units=self.num_targets,
                boundary_casting=self.boundary_casting,
                tf_summary=self.tf_summary,
                dtype=self.policy,
            )
        )

        # return emb_layers, fc_layers
        return relu_layer, fc_layers

    def _build_graph_layers(self):
        return NotImplementedError("To be implemented child class!!")

    def _write_summaries(self, scaler_dict):
        """
        Write summaries to tensorboard for scaler_dict
        """
        for sname, sval in scaler_dict.items():
            tf.compat.v1.summary.scalar(sname, tf.cast(sval, tf.float32))


class ChEMBL20Classifier(GNN):
    """
    GNN model for ChEMBL20 classification
    Includes loss and eval metrics functions common to all ChEMBL20 GNN models
    :param dict params: Model configuration parameters.
    """

    def __init__(self, params):
        super(ChEMBL20Classifier, self).__init__(params)
        self.labels_mask_value = -1

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Return computed loss
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
        ], f"The model supports only TRAIN and EVAL modes."

        labels = tf.cast(labels, tf.float32)
        label_num_valid = features["label_num_valid"]
        loss = self._mean_masked_sigmoid_cross_entropy_loss(
            labels, model_outputs, label_num_valid
        )

        # write summaries for training
        self._write_summaries(
            {'train/cost_masked_cls': loss,}
        )
        return loss

    def build_eval_metric_ops(self, model_outputs, labels):
        """
        Compute metrics and return
        """
        labels = tf.cast(labels, tf.float32)
        masked_weights = self._labels_mask(labels)
        predictions = tf.cast(tf.math.sigmoid(model_outputs), tf.float32)

        auc = dict()
        auc["auroc/overall"] = tf.compat.v1.metrics.auc(
            labels=labels,
            predictions=predictions,
            weights=masked_weights,
            curve='ROC',
            summation_method='trapezoidal',
        )

        per_task_auroc = [
            tf.compat.v1.metrics.auc(
                labels=tf.gather(labels, k, axis=1),
                predictions=tf.gather(predictions, k, axis=1),
                weights=tf.gather(masked_weights, k, axis=1),
                curve='ROC',
                summation_method='trapezoidal',
            )
            for k in range(self.num_targets)
        ]
        valid_task_mask = tf.not_equal(
            tf.reduce_sum(masked_weights, axis=0)
            * tf.reduce_sum(labels * masked_weights, axis=0),
            0.0,
        )
        auc["auroc/batch_mean_per_task"] = tf.compat.v1.metrics.mean(
            tf.stack(per_task_auroc), weights=valid_task_mask
        )

        metrics_dict = {
            "accuracy": tf.compat.v1.metrics.accuracy(
                labels=labels,
                predictions=tf.greater(predictions, 0.5),
                weights=masked_weights,
            ),
            **auc,
        }

        return metrics_dict

    # def build_eval_metric_ops(self, model_outputs, labels):
    #     """
    #     Compute metrics and return
    #     """
    #     labels = tf.cast(labels, tf.float32)
    #     masked_weights = self._labels_mask(labels)
    #     predictions = tf.cast(tf.math.sigmoid(model_outputs), tf.float32)

    #     rmse = dict()
    #     rmse["rmse/overall"] = tf.compat.v1.metrics.root_mean_squared_error(
    #         labels=labels,
    #         predictions=predictions,
    #         weights=masked_weights,
    #     )

    #     per_task_rmse = [
    #         tf.compat.v1.metrics.root_mean_squared_error(
    #             labels=tf.gather(labels, k, axis=1),
    #             predictions=tf.gather(predictions, k, axis=1),
    #             weights=tf.gather(masked_weights, k, axis=1),
    #         )
    #         for k in range(self.num_targets)
    #     ]
    #     valid_task_mask = tf.not_equal(
    #         tf.reduce_sum(masked_weights, axis=0)
    #         * tf.reduce_sum(labels * masked_weights, axis=0),
    #         0.0,
    #     )
    #     rmse["rmse/batch_mean_per_task"] = tf.compat.v1.metrics.mean(
    #         tf.stack(per_task_rmse), weights=valid_task_mask
    #     )

    #     metrics_dict = {
    #         "accuracy": tf.compat.v1.metrics.accuracy(
    #             labels=labels,
    #             predictions=tf.greater(predictions, 0.5),
    #             weights=masked_weights,
    #         ),
    #         **rmse,
    #     }

    #     return metrics_dict

    def _labels_mask(self, labels):
        """
        Creates mask out of labels and gets number of valid
        labels per batch
        """
        # create binary mask to mask out -1 in labels
        mask = tf.not_equal(labels, self.labels_mask_value)
        mask = tf.cast(mask, tf.float32)
        return mask

    def _mean_masked_sigmoid_cross_entropy_loss(
        self, labels, logits, num_valid
    ):
        """
        Modified, actually returns mean RMSE
        """

        # create binary mask
        mask = self._labels_mask(labels)

        # calculate loss and zero out using mask
        _labels = tf.cast(labels, tf.float32)
        _logits = tf.cast(logits, tf.float32)

        # RMSE attempt 1
        # loss = tf.math.squared_difference(_logits, _labels)
        # loss = tf.math.sqrt(loss)

        # RMSE attempt 2
        # loss = tf.sqrt(tf.reduce_mean((_labels - _logits)**2))

        # Original loss
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=_labels, logits=_logits
        # )

        loss = self.squared_error(_labels, _logits)
        loss = tf.sqrt(tf.reduce_mean(loss))

        loss = tf.multiply(loss, mask)

        loss = tf.reduce_mean(
            tf.reduce_sum(loss, axis=1) * tf.cast(num_valid, tf.float32)
        )
        loss = tf.cast(loss, logits.dtype)

        return loss


class GCNModel(ChEMBL20Classifier):
    def __init__(self, params):
        super(GCNModel, self).__init__(params)

    def _build_graph_layers(self):
        graph_layers = []
        for _ in range(self.gnn_depth):
            graph_layers.append(
                GraphConvolutionLayer(
                    in_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    activation=self.activation,
                    pre_activation=False,
                    use_bias=self.use_bias,
                    normalize=True,
                    layer_norm_epsilon=self.layer_norm_epsilon,
                    dropout_rate=self.dropout_rate,
                    boundary_casting=self.boundary_casting,
                    tf_summary=self.tf_summary,
                    dtype=self.policy,
                )
            )

        return graph_layers

    def _call_graph_layers(self, graph_layers, output, adj, is_training):
        for layer in graph_layers:
            output = layer((adj, output), is_training)
        return output


class GATModel(ChEMBL20Classifier):
    def _build_graph_layers(self):
        graph_layers = []
        for i in range(self.gnn_depth):
            if (
                self.attn_reduction == "only_last_layer_mean_reduction"
                and i == self.gnn_depth - 1
            ):
                _concat_heads = False
            else:
                _concat_heads = self.concat_heads

            graph_layers.append(
                GraphAttentionLayer(
                    channels=self.hidden_dim,
                    activation=self.activation,
                    num_heads=self.num_heads,
                    concat_heads=_concat_heads,
                    dropout_rate=self.dropout_rate,
                    layer_norm_epsilon=self.layer_norm_epsilon,
                    boundary_casting=self.boundary_casting,
                    tf_summary=self.tf_summary,
                    dtype=self.policy,
                )
            )
        return graph_layers

    def _call_graph_layers(self, graph_layers, output, adj, is_training):
        for layer in graph_layers:
            output = layer((adj, output), is_training)
        return output