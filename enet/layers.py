# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import tensorflow as tf
import tensorflow.keras.layers as tfl

from .utils import max_unpool_2d


class InitialBlock(tfl.Layer):
    def __init__(self) -> None:
        super().__init__(name='InitialBlock')

        self.zero_pad = tf.keras.layers.ZeroPadding2D()
        self.conv = tfl.Conv2D(13, kernel_size=3, strides=(2, 2))
        self.max_pool = tfl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.concat = tfl.Concatenate()
        self.bn = tfl.BatchNormalization()
        self.activation = tfl.PReLU()

    def call(self, inputs):
        x = self.zero_pad(inputs)
        x = self.conv(x)
        y = self.max_pool(inputs)
        y = self.concat([x, y])
        y = self.bn(y)
        y = self.activation(y)
        return y


class BottleneckBlock(tfl.Layer):
    def __init__(self, major_id, minor_id,
                 output_filters,
                 compression_ratio=4,
                 kernel_size=(3, 3),
                 kernel_strides=(1, 1),
                 padding='same',
                 dilation_rate=(1, 1),
                 dropout_rate=0.1,
                 upsample=False,
                 downsample=False) -> None:
        super().__init__(name=f'Bottleneck{major_id}.{minor_id}')

        if upsample and downsample:
            raise ValueError("Can't upsample and downsample at the same time")
        hidden_filters = max(1, output_filters // compression_ratio)
        # First convolution
        if downsample:
            self.conv1 = tfl.Conv2D(hidden_filters,
                                    kernel_size=(2, 2),
                                    strides=(2, 2),
                                    use_bias=False)
        else:
            self.conv1 = tfl.Conv2D(hidden_filters,
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    use_bias=False)
        self.norm1 = tfl.BatchNormalization()
        self.prelu1 = tfl.PReLU()

        # Main convolution
        # TODO: should be disable bias when asymmetric?
        if upsample:
            self.conv_main = tfl.Conv2DTranspose(hidden_filters,
                                                 kernel_size=kernel_size,
                                                 strides=kernel_strides,
                                                 padding=padding,
                                                 dilation_rate=dilation_rate,
                                                 use_bias=True)
            self.is_asymmetric = False
            self.skip_conv = tfl.Conv2D(output_filters, 1, use_bias=False)
            self.skip_bn = tfl.BatchNormalization()
        else:
            self.conv_main = tfl.Conv2D(hidden_filters,
                                        kernel_size=kernel_size,
                                        strides=kernel_strides,
                                        padding=padding,
                                        dilation_rate=dilation_rate,
                                        use_bias=True)
            self.is_asymmetric = kernel_size[0] != kernel_size[1]
            if self.is_asymmetric:
                # TODO: should we place BatchNormalization + PReLU between
                # the two asymmetric convolutions?
                self.conv_main2 = tfl.Conv2D(hidden_filters,
                                             kernel_size=kernel_size[::-1],
                                             strides=kernel_strides[::-1],
                                             padding=padding,
                                             dilation_rate=dilation_rate,
                                             use_bias=True)
        self.bn_main = tfl.BatchNormalization()
        self.activation_main = tfl.PReLU()

        # Final convolution
        self.conv_out = tf.keras.layers.Conv2D(output_filters,
                                               kernel_size=(1, 1),
                                               strides=(1, 1),
                                               use_bias=False)
        self.bn_out = tfl.BatchNormalization()
        self.activation_out = tfl.PReLU()

        # Regularizer
        self.dropout_out = tfl.SpatialDropout2D(dropout_rate)

        # Skip branch
        self.upsample = upsample
        self.downsample = downsample
        self.output_filters = output_filters

        self.res_add = tfl.Add()
        self.res_activation = tfl.PReLU()

    def call(self, inputs):
        if isinstance(inputs, dict):
            assert 'input' in inputs
            x = inputs['input']
            y = inputs['input']
            assert 'indices' in inputs
            indices = inputs['indices']
        else:
            x = inputs
            y = inputs
            indices = None
        x_shape = x.shape
        in_channels = x_shape[-1]
        # Evaluate first convolution
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.prelu1(x)

        # Evaluate main convolution
        x = self.conv_main(x)
        if self.is_asymmetric:
            x = self.conv_main2(x)
        x = self.bn_main(x)
        x = self.activation_main(x)

        # Evaluate final convolution
        x = self.conv_out(x)
        x = self.bn_out(x)
        x = self.activation_out(x)
        x = self.dropout_out(x)

        if self.upsample:
            assert indices is not None
            indices_shape = indices.shape
            y = self.skip_conv(y)
            y = self.skip_bn(y)
            if x_shape[1:] != indices_shape[1:]:
                assert indices_shape[1] >= x_shape[1]
                assert indices_shape[2] >= x_shape[2]
                y = tf.pad(y,
                           tf.constant([[0, 0],
                                        [indices_shape[1] - x_shape[1], 0],
                                        [indices_shape[2] - x_shape[2], 0],
                                        [0, 0]]),
                           'CONSTANT', 0.0)
            y = max_unpool_2d(y, indices=indices, kernel_size=2, strides=2)
        elif self.downsample:
            y, argmax = tf.nn.max_pool_with_argmax(y,
                                                   ksize=(1, 2, 2, 1),
                                                   strides=(1, 2, 2, 1),
                                                   padding='VALID')

        skip_channels_pad = self.output_filters - in_channels
        if skip_channels_pad > 0:
            y = tf.pad(y,
                       tf.constant([[0, 0],
                                    [0, 0],
                                    [0, 0],
                                    [0, skip_channels_pad]]),
                       'CONSTANT', 0.0)

        if y.shape[1:] != x.shape[1:]:
            assert y.shape[1] >= x.shape[1]
            assert y.shape[2] >= x.shape[2]
            x = tf.pad(x,
                       tf.constant([[0, 0],
                                    [y.shape[1] - x.shape[1], 0],
                                    [y.shape[2] - x.shape[2], 0],
                                    [0, 0]]),
                       'CONSTANT', 0.0)

        x = self.res_add([x, y])
        x = self.res_activation(x)

        if self.downsample:
            return x, argmax
        else:
            return x


class DilatedAsymmetricBottleneckBlock(tfl.Layer):
    def __init__(self, major_id, start_minor_id,
                 output_filters,
                 compression_ratio=4,
                 base_dilation_rate=2,
                 dropout_rate=0.1):
        super().__init__(
            name=f'DilatedAsymmetricBottleneck{major_id}.{start_minor_id}')

        self.bn1 = BottleneckBlock(major_id, start_minor_id+0,
                                   output_filters=output_filters,
                                   compression_ratio=compression_ratio,
                                   kernel_size=(3, 3),
                                   kernel_strides=(1, 1),
                                   padding='same',
                                   dilation_rate=1,
                                   dropout_rate=dropout_rate)
        self.bn2 = BottleneckBlock(major_id, start_minor_id+1,
                                   output_filters=output_filters,
                                   compression_ratio=compression_ratio,
                                   kernel_size=(3, 3),
                                   kernel_strides=(1, 1),
                                   padding='same',
                                   dilation_rate=base_dilation_rate,
                                   dropout_rate=dropout_rate)
        self.bn3 = BottleneckBlock(major_id, start_minor_id+2,
                                   output_filters=output_filters,
                                   compression_ratio=compression_ratio,
                                   kernel_size=(5, 1),
                                   kernel_strides=(1, 1),
                                   padding='same',
                                   dilation_rate=1,
                                   dropout_rate=dropout_rate)
        self.bn4 = BottleneckBlock(major_id, start_minor_id+3,
                                   output_filters=output_filters,
                                   compression_ratio=compression_ratio,
                                   kernel_size=(3, 3),
                                   kernel_strides=(1, 1),
                                   padding='same',
                                   dilation_rate=2*base_dilation_rate,
                                   dropout_rate=dropout_rate)

    def call(self, inputs):
        x = inputs
        x = self.bn1(x)
        x = self.bn1(x)
        x = self.bn1(x)
        x = self.bn1(x)
        return x
