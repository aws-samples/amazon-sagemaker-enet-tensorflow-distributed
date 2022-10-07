# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import tensorflow as tf
import collections.abc
import numpy as np


class SMMetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, prefix='###'):
        super().__init__()
        self.prefix = prefix
        self.keys = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return f'"[{", ".join(map(str, k))}]"'
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict((k, logs[k]) if k in logs else (k, 'NA')
                        for k in self.keys)

        metrics_dict = {'epoch': epoch}
        metrics_dict.update(
            (key, handle_value(logs[key])) for key in self.keys)
        for k, v in metrics_dict.items():
            print(f'{self.prefix} {k} = {v}')
        print(f'{self.prefix} -------------------------')
