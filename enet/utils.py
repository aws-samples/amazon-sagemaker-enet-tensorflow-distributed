# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import tensorflow as tf
from keras.utils import conv_utils


def _calculate_max_unpool_2d_output_shape(input_shape, pool_size, strides, padding):
    if padding == 'valid':
        output_shape = (
            input_shape[0],
            (input_shape[1] - 1) * strides[0] + pool_size[0],
            (input_shape[2] - 1) * strides[1] + pool_size[1],
            input_shape[3],
        )
    elif padding == 'same':
        output_shape = (
            input_shape[0],
            input_shape[1] * strides[0],
            input_shape[2] * strides[1],
            input_shape[3],
        )
    else:
        raise ValueError('Padding must be a string from: "same", "valid"')
    return output_shape


def max_unpool_2d(x, indices, kernel_size, strides=(1, 1), padding='valid'):
    rank = 2
    kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    strides = conv_utils.normalize_tuple(
        strides, rank, 'strides')

    indices = tf.cast(indices, 'int32')
    input_shape_tensor = tf.shape(x, out_type='int32')
    input_shape = x.get_shape().as_list()
    output_shape_tensor = _calculate_max_unpool_2d_output_shape(
        input_shape_tensor, kernel_size, strides, padding)
    output_shape = _calculate_max_unpool_2d_output_shape(
        input_shape, kernel_size, strides, padding)

    # Calculates indices for batch, height, width and feature maps.
    one_like_mask = tf.ones_like(indices, dtype="int32")
    batch_shape = tf.concat([[input_shape_tensor[0]], [1], [1], [1]], axis=0)
    batch_range = tf.reshape(
        tf.range(output_shape_tensor[0], dtype="int32"), shape=batch_shape
    )
    b_ = one_like_mask * batch_range
    y_ = indices // (output_shape_tensor[2] * output_shape_tensor[3])
    x_ = (indices // output_shape_tensor[3]) % output_shape_tensor[2]
    feature_range = tf.range(output_shape_tensor[3], dtype="int32")
    f_ = one_like_mask * feature_range

    # Transposes indices & reshape update values to one dimension.
    updates_size = tf.size(x)
    indices = tf.transpose(tf.reshape(
        tf.stack([b_, y_, x_, f_]), [4, updates_size]))
    values = tf.reshape(x, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape_tensor)
    ret.set_shape(output_shape)
    return ret
