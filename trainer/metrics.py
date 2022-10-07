# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import tensorflow as tf


class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self,
                 num_classes,
                 name=None,
                 dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.class_idxs = tf.range(0, num_classes)
        self.intersections = self.add_weight(name='iou_intersections',
                                             shape=(num_classes,),
                                             dtype=dtype,
                                             initializer='zeros')
        self.unions = self.add_weight(name='iou_unions',
                                      shape=(num_classes,),
                                      dtype=dtype,
                                      initializer='zeros')

    @tf.function
    def _iu_by_class(self, y_true, y_pred, class_idx):
        y_true_mask = tf.equal(y_true, class_idx)
        y_pred_mask = tf.equal(y_pred, class_idx)
        intersection_mask = tf.logical_and(y_true_mask, y_pred_mask)
        union_mask = tf.logical_or(y_true_mask, y_pred_mask)
        intersection = tf.cast(intersection_mask, tf.float32)
        union = tf.cast(union_mask, tf.float32)
        intersection_sum = tf.reduce_sum(intersection, axis=None)
        union_sum = tf.reduce_sum(union, axis=None)
        iu = tf.stack([intersection_sum, union_sum])
        return iu

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, self.class_idxs.dtype)
        y_pred = tf.cast(y_pred, self.class_idxs.dtype)
        is_and_us = tf.map_fn(
            lambda idx: self._iu_by_class(y_true, y_pred, idx),
            self.class_idxs,
            fn_output_signature=tf.float32)
        self.intersections.assign_add(is_and_us[:, 0])
        self.unions.assign_add(is_and_us[:, 1])

    def result(self):
        iou = tf.math.divide(self.intersections, self.unions)
        iou = tf.where(tf.math.is_nan(iou), tf.zeros_like(iou), iou)
        iou = tf.divide(
            tf.reduce_sum(iou),
            tf.cast(iou.shape[0], iou.dtype),
        )
        return iou

    def reset_state(self):
        self.intersections.assign(tf.zeros_like(self.intersections))
        self.unions.assign(tf.zeros_like(self.unions))

    def get_config(self):
        return {'num_classes': self.num_classes}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
