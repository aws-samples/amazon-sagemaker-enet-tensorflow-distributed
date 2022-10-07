# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from dataclasses import dataclass
from typing import Tuple

import tensorflow.keras as tfk
import tensorflow.keras.layers as tfl

from .layers import (
    InitialBlock,
    BottleneckBlock,
    DilatedAsymmetricBottleneckBlock,
)


@dataclass
class ENetParams:
    input_dim: Tuple[int, int, int]  # H x W x C
    num_object_classes: int
    dropout_rate1: float
    dropout_rate2: float


class ENetModel(tfk.Model):
    def __init__(self, params: ENetParams):
        super().__init__(self, name='ENet')
        self.input_dim = params.input_dim

        # Initial block
        self.init = InitialBlock()

        # First bottleneck block
        self.bn1 = [
            BottleneckBlock(1, 1,
                            output_filters=64,
                            downsample=True,
                            dropout_rate=params.dropout_rate1)
        ]
        for i in range(4):
            self.bn1.append(
                BottleneckBlock(1, i+2,
                                output_filters=64,
                                dropout_rate=params.dropout_rate1)
            )

        # Second bottleneck block
        self.bn2 = [
            BottleneckBlock(2, 1,
                            output_filters=128,
                            downsample=True,
                            dropout_rate=params.dropout_rate2),
            DilatedAsymmetricBottleneckBlock(2, 2,
                                             output_filters=128,
                                             base_dilation_rate=2,
                                             dropout_rate=params.dropout_rate2),
            DilatedAsymmetricBottleneckBlock(2, 6,
                                             output_filters=128,
                                             base_dilation_rate=8,
                                             dropout_rate=params.dropout_rate2),
        ]

        # Third bottleneck block
        self.bn31 = DilatedAsymmetricBottleneckBlock(3, 1,
                                                     output_filters=128,
                                                     base_dilation_rate=2,
                                                     dropout_rate=params.dropout_rate2)
        self.bn32 = DilatedAsymmetricBottleneckBlock(3, 5,
                                                     output_filters=128,
                                                     base_dilation_rate=8,
                                                     dropout_rate=params.dropout_rate2)

        # Fourth bottleneck block
        self.bn4 = [
            BottleneckBlock(4, 1,
                            output_filters=64,
                            upsample=True,
                            dropout_rate=params.dropout_rate2)
        ]
        for i in range(2):
            self.bn4.append(
                BottleneckBlock(4, i+2,
                                output_filters=64,
                                dropout_rate=params.dropout_rate2)
            )

        # Fifth bottleneck block
        self.bn51 = BottleneckBlock(5, 1,
                                    output_filters=16,
                                    upsample=True,
                                    dropout_rate=params.dropout_rate2)
        self.bn52 = BottleneckBlock(5, 2,
                                    output_filters=16,
                                    dropout_rate=params.dropout_rate2)

        # Full convolution
        self.fullconv = tfl.Conv2DTranspose(params.num_object_classes,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            use_bias=False)
        self.softmax = tfl.Softmax()

    def call(self, inputs):
        x = self.init(inputs)

        x, indices1 = self.bn1[0](x)
        for bn in self.bn1[1:]:
            x = bn(x)

        x, indices2 = self.bn2[0](x)
        for bn in self.bn2[1:]:
            x = bn(x)

        x = self.bn31(x)
        x = self.bn32(x)

        x = self.bn4[0]({'input': x, 'indices': indices2})
        for bn in self.bn4[1:]:
            x = bn(x)

        x = self.bn51({'input': x, 'indices': indices1})
        x = self.bn52(x)

        x = self.fullconv(x)
        x = self.softmax(x)
        return x
