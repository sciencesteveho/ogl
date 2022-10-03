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


class GraphConvolutionLayer(BaseLayer):
    """Implementation of Cerebras layer for GraphConvolution.

    **Reference**:

        - `Kipf & Welling 2016 <https://arxiv.org/abs/1609.02907>`_.


    **Note**: One difference from the Kipf & Welling paper is that
    this class applies the activation function before the adjacency matrix
    multiplication. This class provides the ``pre_activation`` parameter to
    enable the activation function before or after the adjacency
    multiplication. If set to ``False``, the layer applies activation after
    adjacency multiplication.

    Args:
        in_dim (int): Input dimension of the convolution.
        out_dim (int): Output dimension of the convolution.
        activation (callable): Keras Activation to use.
        pre_activation (bool): Specifies whether to apply the activation
            before adjacency multiplication. Defaults to ``True``.
        use_bias (bool): Specifies whether to add bias in the training.
            Defaults to ``True``.
        use_film (bool): Specifies whether to use FiLM in the training.
            Defaults to ``False``.
        normalize (bool): Specifies whether to apply the layer normalization
            directly after adjacency multiplication. Defaults to ``False``.
        layer_norm_epsilon (float): Epsilon value for the layer normalization.
            Defaults to ``1.0e-5``.
        kernel_initializer (str): Keras kernel initializer to use. Defaults to
            ``"glorot_uniform"``.
        bias_initializer (str): Kernel bias initializer to use. Defaults to
            ``"zeros"``.
        boundary_casting (bool): See the documentation for ``BaseLayer``.
        tf_summary: See documentation for ``BaseLayer``.
        **kwargs: Additional keyword arguments for ``BaseLayer``.

    """

    def __init__(
        self,
        in_dim,
        out_dim,
        activation=None,
        pre_activation=True,
        use_bias=True,
        normalize=False,
        layer_norm_epsilon=1.0e-5,
        dropout_rate=0.0,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(GraphConvolutionLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_dropout = dropout_rate > 0.0
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.pre_activation = pre_activation
        self.normalize = normalize
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        # get activations and initializers
        kernel_initializer = tf.compat.v1.keras.initializers.get(
            self.kernel_initializer
        )
        bias_initializer = tf.compat.v1.keras.initializers.get(
            self.bias_initializer
        )

        self.activation_layer = ActivationLayer(
            activation=self.activation,
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.dtype_policy,
        )

        if self.normalize:
            self.norm_layer = LayerNormalizationLayer(
                epsilon=self.layer_norm_epsilon,
                boundary_casting=self.boundary_casting,
                tf_summary=self.tf_summary,
                dtype=self.dtype_policy,
            )
        else:
            self.norm_layer = None
        if self.use_dropout:
            self.dropout_layer = DropoutLayer(
                rate=self.dropout_rate,
                boundary_casting=self.boundary_casting,
                tf_summary=self.tf_summary,
                dtype=self.dtype_policy,
            )
        else:
            self.dropout_layer = None

        # initialize convolution weights and bias
        self.kernel = self.add_weight(
            name='kernel',
            shape=[self.in_dim, self.out_dim],
            dtype=self.variable_dtype,
            experimental_autocast=False,
            initializer=kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[self.out_dim,],
                dtype=self.variable_dtype,
                experimental_autocast=False,
                initializer=bias_initializer,
                trainable=True,
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, training=True, **kwargs):
        """Apply graph convolution to the inputs.

        Args:
            inputs (tuple): Contains the feature matrix of the shape
                ``[batch_size, num_nodes, in_dim]`` and the adjacency matrix of
                the shape ``[batch_size, num_nodes, num_nodes]``.
            **kwargs: Additional keyword arguments for the call argument.

        Returns:
            Graph Convolution layer output of shape
                ``[batch_size, num_nodes, out_dim]``.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)

        adj, feat = inputs

        output = tf.matmul(feat, tf.cast(self.kernel, feat.dtype))
        if self.use_bias:
            output = output + tf.cast(self.bias, output.dtype)
        if self.pre_activation:
            output = self.activation_layer(output)

        output = tf.matmul(adj, output)
        if self.normalize:
            output = self.norm_layer(output)
        if not self.pre_activation:
            output = self.activation_layer(output)
        if self.use_dropout:
            output = self.dropout_layer(output, training)

        if self.tf_summary:
            output = summary_layer(output)
            
        return output