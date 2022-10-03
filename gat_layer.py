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

from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.common.tf.layers.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class _GraphAttentionHead(BaseLayer):
    """Implementation of a single attention head, for use in the rank 2 Graph
    Attention layer.

    **Reference**:

        - `Velickovic et al. (2017) <https://arxiv.org/abs/1710.10903>`_.\


    Args:
        channels (int): Output channels of convolution.
        activation (callable): Keras Activation to use.
        use_bias (bool): Specifies whether to add bias in training.
            Defaults to ``True``.
        dropout_rate (float): Internal dropout rate for attention coefficients.
        neg_inf (float): Negative infinity for masking. Defaults to
            ``-1e4``.
        leaky_relu_alpha (float):  Negative slope coefficient. Defaults to
            ``0.2``.
        kernel_initializer (str): Keras kernel initializer to use. Defaults to
            ``"glorot_uniform"``.
        bias_initializer (str): Kernel bias initializer to use. Defaults to
            ``"zeros"``.
        attn_kernel_initializer (str): Keras kernel initializer to use for
            attention. Defaults to ``"glorot_uniform"``.
        boundary_casting (bool): See documentation for ``BaseLayer``.
        tf_summary: See documentation for ``BaseLayer``.
        **kwargs: Additional keyword arguments for ``BaseLayer``.

    """

    def __init__(
        self,
        channels,
        activation=None,
        use_bias=True,
        dropout_rate=0.0,
        neg_inf=-1e4,
        leaky_relu_alpha=0.2,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(_GraphAttentionHead, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.channels = channels
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.attn_kernel_initializer = attn_kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        assert neg_inf < -100, f"neg_inf is {neg_inf}, it should be smaller"
        self.neg_inf = neg_inf
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_rate = dropout_rate
        self.dropout = True if self.dropout_rate > 0.0 else False

    def build(self, input_shape):
        # get input and output shapes
        assert len(input_shape) >= 2
        in_dim = input_shape[1][-1]

        # get activations and initializers
        kernel_initializer = tf.compat.v1.keras.initializers.get(
            self.kernel_initializer
        )
        bias_initializer = tf.compat.v1.keras.initializers.get(
            self.bias_initializer
        )
        attn_kernel_initializer = tf.compat.v1.keras.initializers.get(
            self.attn_kernel_initializer
        )

        # define weights, biases and attention weights
        self.kernel = self.add_weight(
            name="kernel",
            shape=[in_dim, self.channels],
            dtype=self.variable_dtype,
            experimental_autocast=False,
            initializer=kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.channels,],
                dtype=self.variable_dtype,
                experimental_autocast=False,
                initializer=bias_initializer,
                trainable=True,
            )
        else:
            self.bias = None
        self.attn_kernel_self = self.add_weight(
            name="attn_kernel_self",
            shape=[self.channels, 1],
            dtype=self.variable_dtype,
            experimental_autocast=False,
            initializer=attn_kernel_initializer,
            trainable=True,
        )
        self.attn_kernel_neigh = self.add_weight(
            name="attn_kernel_neigh",
            shape=[self.channels, 1],
            dtype=self.variable_dtype,
            experimental_autocast=False,
            initializer=attn_kernel_initializer,
            trainable=True,
        )

        if self.dropout:
            self.dropout_layer = DropoutLayer(
                rate=self.dropout_rate, dtype=self.dtype_policy
            )
        else:
            self.dropout_layer = None

        self.built = True

    def call(self, inputs, training=True, **kwargs):
        """Apply a single attention head to the inputs.

        Args:
            inputs (tuple): Contains adjacency matrix of the shape
                ``[batch_size, num_nodes, num_nodes]`` and the feature matrix
                of the shape ``[batch_size, num_nodes, in_dim]``.
            **kwargs: Additional keyword arguments for the call argument.

        Returns:
            Tensor: A tensor of shape ``[batch_size, num_nodes, channels]``.
        """

        adj, feat = inputs
        feat = self._attend(adj, feat, training=training)
        return feat

    def _attend(self, adj, feat, training):
        feat = tf.matmul(feat, tf.cast(self.kernel, feat.dtype))
        feat = (
            feat + tf.cast(self.bias, feat.dtype)
            if self.bias is not None
            else feat
        )
        attn_self = tf.matmul(feat, tf.cast(self.attn_kernel_self, feat.dtype))
        attn_neigh = tf.matmul(
            feat, tf.cast(self.attn_kernel_neigh, feat.dtype)
        )
        attn_shape = attn_neigh.get_shape().as_list()
        shape = attn_shape[:-2] + [attn_shape[-1]] + [attn_shape[-2]]
        shape = [d if d is not None else -1 for d in shape]
        attn_neigh = tf.reshape(attn_neigh, shape)
        e = attn_self + attn_neigh

        mask = self.neg_inf - self.neg_inf * adj
        e = tf.compat.v1.nn.leaky_relu(e, alpha=self.leaky_relu_alpha)
        e += mask

        attn = tf.compat.v1.nn.softmax(e, axis=-1)
        if self.dropout:
            attn = self.dropout_layer(attn, training)

        feat = tf.matmul(attn, feat)

        return feat


class GraphAttentionLayer(BaseLayer):
    """Implementation of the Cerebras layer GraphAttention.

    **Reference**:

        - `Velickovic et al. (2017) <https://arxiv.org/abs/1710.10903>`_.


    Args:
        channels (int): Output channels of convolution.
        activation (callable): Keras Activation to use.
        normalize_output (bool): Specifies whether to normalize outputs of
            graph attention. Defaults to ``True``.
        use_bias (bool): Specifies whether to add bias in training.
            Defaults to ``True``.
        num_heads (int): Number of attention heads to use. Defaults to ``1``.
        concat_heads (bool): Specifies whether to concatenate the output of
            the attention heads instead of averaging. Defaults to ``True``.
        dropout_rate (float): Internal dropout rate for attention coefficients.
        layer_norm_epsilon (float): Espilon value to be used for layer norm.
        neg_inf (float): Negative infinity for masking. Defaults to ``-1e4``.
        leaky_relu_alpha (float): Negative slope coefficient. Defaults to
            ``0.2``.
        kernel_initializer (str): Keras kernel initializer to use. Defaults to
            ``"glorot_uniform"``.
        bias_initializer (str): Kernel bias initializer to use. Defaults to
            ``"zeros"``.
        attn_kernel_initializer(str): Keras kernel initializer to use for
            attention. Defaults to ``"glorot_uniform"``.
        boundary_casting (bool): See documentation for ``BaseLayer``.
        tf_summary: See documentation for ``BaseLayer``.
        **kwargs: Additional keyword arguments for ``BaseLayer``.

    """

    def __init__(
        self,
        channels,
        activation=None,
        normalize_output=True,
        use_bias=True,
        num_heads=1,
        concat_heads=True,
        dropout_rate=0.5,
        layer_norm_epsilon=1e-5,
        neg_inf=-1e4,
        leaky_relu_alpha=0.2,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(GraphAttentionLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.channels = channels
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.attn_kernel_initializer = attn_kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.concat_heads = concat_heads
        self.normalize_output = normalize_output
        assert neg_inf < -100, f"neg_inf is {neg_inf}, it should be smaller"
        self.neg_inf = neg_inf
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon

        self.dropout = True if self.dropout_rate > 0.0 else False

    def build(self, input_shape):
        # get input and output shapes
        assert len(input_shape) >= 2
        in_dim = input_shape[1][-1]

        if self.concat_heads:
            self.output_dim = self.channels * self.num_heads
        else:
            self.output_dim = self.channels

        self.attention_heads = []
        for i in range(self.num_heads):
            self.attention_heads.append(
                _GraphAttentionHead(
                    name=f"attn_head_{i}",
                    channels=self.channels,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    dropout_rate=self.dropout_rate,
                    neg_inf=self.neg_inf,
                    leaky_relu_alpha=self.leaky_relu_alpha,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    attn_kernel_initializer=self.attn_kernel_initializer,
                    dtype=self.dtype_policy,
                )
            )

        self.activation_layer = ActivationLayer(
            activation=self.activation,
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.dtype_policy,
        )

        if self.normalize_output:
            self.norm_layer = LayerNormalizationLayer(
                epsilon=self.layer_norm_epsilon, dtype=self.dtype_policy
            )
        else:
            self.norm_layer = None

        self.built = True

    def call(self, inputs, training=True, **kwargs):
        """Apply graph attention to inputs.

        Args:
            inputs (tuple): Contains adjacency matrix of the shape
                ``[batch_size, num_nodes, num_nodes]`` and feature matrix of
                the shape ``[batch_size, num_nodes, in_dim]``.
            **kwargs: Additional keyword arguments for the call argument.

        Returns:
            Graph Attention layer output of shape
            ``[batch_size, num_nodes, channels]``.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)

        outputs = [h(inputs, training) for h in self.attention_heads]

        outputs = self._aggregate(outputs)
        outputs = self._activate(outputs)
        if self.normalize_output:
            outputs = self.norm_layer(outputs)

        if self.tf_summary:
            outputs = summary_layer(outputs)

        return outputs

    def _aggregate(self, feat):
        if self.concat_heads:
            output = tf.concat(feat, axis=-1)
        else:
            output = tf.stack(feat, axis=-1)
            output = tf.reduce_mean(output, axis=-1)
        return output

    def _activate(self, feat):
        return self.activation_layer(feat)