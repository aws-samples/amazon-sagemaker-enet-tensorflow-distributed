# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import tensorflow as tf

from trainer.metrics import MeanIoU

import smdistributed.dataparallel.tensorflow as sdp


@dataclass
class TrainerParams:
    num_classes: int
    batch_size: int
    learning_rate: float
    epochs: int
    save_checkpoint: bool = False
    checkpoint_path: Optional[Path] = None
    load_best_checkpoint_after_fit: bool = False


class Trainer:
    def __init__(self, params: TrainerParams,
                 dataset_loader: Any,
                 model: tf.keras.Model,
                 extra_callbacks: List[tf.keras.callbacks.Callback] = []):
        self.params = params
        self.loader = dataset_loader
        self.model = model
        self.callbacks = extra_callbacks

        self._build_optimizer()
        self._build_loss()
        self._build_metrics()
        self._prepare_model()
        if self.params.save_checkpoint:
            assert self.params.checkpoint_path is not None
            self._build_checkpoint_callback()

    def _build_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.params.learning_rate * sdp.size(),
        )

    def _build_loss(self):
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def _build_metrics(self):
        self.metrics = [
            MeanIoU(self.params.num_classes, name='mean_iou')
        ]

    def _prepare_model(self):
        pass

    def _build_checkpoint_callback(self):
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.params.checkpoint_path / 'ckpt',
                save_best_only=True,
                save_weights_only=True,
                save_freq='epoch',
            )
        )

    def _reset_metric_states(self):
        for metric in self.metrics:
            metric.reset_state()

    @tf.function
    def _get_metric_results(self, validation: bool) -> Dict[str, float]:
        prefix = 'val_' if validation else ''
        return {f'{prefix}{m.name}': sdp.oob_allreduce(m.result())
                for m in self.metrics}

    @tf.function
    def _train_step(self, X, y, first_batch):
        with tf.GradientTape() as tape:
            output = self.model(X, training=True)
            loss_value = self.loss(y, output)

        for metric in self.metrics:
            metric.update_state(y, output)

        # SMDataParallel: Wrap tf.GradientTape with SMDataParallel's DistributedGradientTape
        tape = sdp.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        if first_batch:
            # SMDataParallel: Broadcast model and optimizer variables
            sdp.broadcast_variables(self.model.variables, root_rank=0)
            sdp.broadcast_variables(self.optimizer.variables(), root_rank=0)

        # SMDataParallel: all_reduce call
        # Average the loss across workers
        loss_value = sdp.oob_allreduce(loss_value)
        return loss_value

    @tf.function
    def _val_step(self, X, y):
        output = self.model(X, training=False)

        for metric in self.metrics:
            metric.update_state(y, output)

        loss_value = self.loss(y, output)
        # SMDataParallel: all_reduce call
        # Average the loss across workers
        loss_value = sdp.oob_allreduce(loss_value)
        return loss_value

    def fit(self):
        if self.params.checkpoint_path is not None and self.params.checkpoint_path.exists():
            print(f'Loading from checkpoint at {self.params.checkpoint_path}')
            checkpoint = tf.train.Checkpoint(self.model)
            checkpoint.restore(
                str(self.params.checkpoint_path / 'ckpt')).expect_partial()
        train_ds = self.loader.train\
            .prefetch(tf.data.AUTOTUNE)\
            .shuffle(self.params.batch_size * 512)\
            .batch(self.params.batch_size)
        val_ds = self.loader.val.prefetch(
            tf.data.AUTOTUNE).batch(self.params.batch_size)

        callbacks = tf.keras.callbacks.CallbackList(
            self.callbacks, model=self.model)

        if sdp.rank() == 0:
            callbacks.on_train_begin()
        for epoch in range(self.params.epochs):
            metric_values = {}

            if sdp.rank() == 0:
                callbacks.on_epoch_begin(epoch, metric_values)

            # Training steps
            train_loss_sum = 0.0
            train_loss_count = 0
            for step, (X_train, y_train) in enumerate(train_ds):
                first_batch = step == 0
                loss_value = self._train_step(X_train, y_train, first_batch)
                train_loss_sum += float(loss_value)
                train_loss_count += 1
            train_loss = train_loss_sum / float(train_loss_count)

            metric_values['loss'] = train_loss

            metric_values.update(self._get_metric_results(False))
            self._reset_metric_states()

            # Validation steps
            val_loss_sum = 0.0
            val_loss_count = 0
            for X_val, y_val in val_ds:
                loss_value = self._val_step(X_val, y_val)
                val_loss_sum += float(loss_value)
                val_loss_count += 1
            val_loss = val_loss_sum / float(val_loss_count)

            metric_values['val_loss'] = val_loss

            metric_values.update(self._get_metric_results(True))
            self._reset_metric_states()

            if sdp.rank() == 0:
                callbacks.on_epoch_end(epoch, metric_values)
        if sdp.rank() == 0:
            callbacks.on_train_end()

        if sdp.rank() == 0 and self.params.save_checkpoint and self.params.load_best_checkpoint_after_fit:
            checkpoint = tf.train.Checkpoint(self.model)
            checkpoint.restore(
                str(self.params.checkpoint_path / 'ckpt')).expect_partial()

    def save_model(self, model_path: Path):
        if sdp.rank() == 0:
            self.model.save(model_path / '1',
                            overwrite=True)
