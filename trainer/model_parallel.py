# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf

from trainer.metrics import MeanIoU

import smdistributed.modelparallel.tensorflow as smp

from enet.model import ENetParams, ENetModel


class DistributedENet(smp.DistributedModel):
    def __init__(self, params: ENetParams):
        super().__init__(name='ENet')
        self.enet = ENetModel(params)

    def call(self, inputs):
        return self.enet(inputs)


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
                 model: DistributedENet,
                 input_dim: Tuple[int, int, int],
                 extra_callbacks: List[tf.keras.callbacks.Callback] = []):
        self.params = params
        self.input_dim = input_dim
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
        with tf.name_scope('optimizer'):
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.params.learning_rate,
            )

    def _build_loss(self):
        with tf.name_scope('loss'):
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
        return {f'{prefix}{m.name}': m.result() for m in self.metrics}

    # smdistributed: Define smp.step. Return any tensors needed outside
    @smp.step(non_split_inputs=['self'])
    def _get_grads(self, X, y):
        output = self.model(X, training=True)
        loss = self.loss(y, output)
        grads = self.optimizer.get_gradients(
            loss, self.model.trainable_variables)
        return grads, loss, output

    @smp.step(non_split_inputs=['self'])
    def _get_val_loss(self, X, y):
        output = self.model(X, training=False)
        loss = self.loss(y, output)

        return loss, output

    @tf.function
    def _train_step(self, X, y):
        gradients, loss_value, output = self._get_grads(X, y)

        # smdistributed: Accumulate the gradients across microbatches
        gradients = [g.accumulate() for g in gradients]
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        output = output.merge()
        for metric in self.metrics:
            metric.update_state(y, output)

        loss_value = loss_value.reduce_mean()
        return loss_value

    @tf.function
    def _val_step(self, X, y):
        loss_value, output = self._get_val_loss(X, y)

        output = output.merge()
        for metric in self.metrics:
            metric.update_state(y, output)

        loss_value = loss_value.reduce_mean()
        return loss_value

    def fit(self):
        train_ds = self.loader.train\
            .prefetch(tf.data.AUTOTUNE)\
            .shuffle(self.params.batch_size * 512, seed=smp.dp_rank())\
            .batch(self.params.batch_size, drop_remainder=True)
        val_ds = self.loader.val.prefetch(
            tf.data.AUTOTUNE).batch(self.params.batch_size, drop_remainder=True)

        if smp.rank() == 0:
            callbacks = tf.keras.callbacks.CallbackList(
                self.callbacks, model=self.model)

        if smp.rank() == 0:
            callbacks.on_train_begin()
        for epoch in range(self.params.epochs):
            metric_values = {}

            if smp.rank() == 0:
                callbacks.on_epoch_begin(epoch, metric_values)

            # Training steps
            train_loss_sum = 0.0
            train_loss_count = 0
            for step, (X_train, y_train) in enumerate(train_ds):
                first_batch = step == 0
                loss_value = self._train_step(X_train, y_train)
                train_loss_sum += float(loss_value)
                train_loss_count += 1
                if epoch == 0 and first_batch:
                    if self.params.checkpoint_path is not None and self.params.checkpoint_path.exists():
                        print(
                            f'Loading from checkpoint at {self.params.checkpoint_path}')
                        checkpoint = smp.CheckpointManager(self.model)
                        checkpoint.restore(
                            str(self.params.checkpoint_path / 'ckpt')).expect_partial()
            if smp.rank() == 0:
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

            if smp.rank() == 0:
                val_loss = val_loss_sum / float(val_loss_count)

                metric_values['val_loss'] = val_loss

                metric_values.update(self._get_metric_results(True))
                self._reset_metric_states()

            if smp.rank() == 0:
                callbacks.on_epoch_end(epoch, metric_values)
        if smp.rank() == 0:
            callbacks.on_train_end()

        if smp.rank() == 0 and self.params.save_checkpoint and self.params.load_best_checkpoint_after_fit:
            checkpoint = smp.CheckpointManager(self.model)
            checkpoint.restore(
                str(self.params.checkpoint_path / 'ckpt'))

    def save_model(self, model_path: Path):
        if smp.rank() == 0:
            checkpoint_path = model_path / '1'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ckpt = tf.train.Checkpoint(
                optimizer=self.optimizer, model=self.model)
            ckpt_manager = tf.train.CheckpointManager(
                ckpt, str(checkpoint_path), max_to_keep=1)
            ckpt_manager.save()
