# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
import tensorflow as tf

from trainer.metrics import MeanIoU


@dataclass
class TrainerParams:
    num_classes: int
    batch_size: int
    learning_rate: float
    epochs: int
    save_checkpoint: bool = False
    checkpoint_path: Optional[Path] = None
    load_best_checkpoint_after_fit: bool = False
    fit_verbosity: int = 0


class Trainer:
    def __init__(self, params: TrainerParams,
                 dataset_loader: Any,
                 model: tf.keras.Model,
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
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.params.learning_rate,
        )

    def _build_loss(self):
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def _build_metrics(self):
        self.metrics = [
            MeanIoU(self.params.num_classes, name='mean_iou')
        ]

    def _prepare_model(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )
        self.model.build((None,) + self.input_dim)
        print(self.model.summary())

    def _build_checkpoint_callback(self):
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.params.checkpoint_path / 'ckpt',
                save_best_only=True,
                save_weights_only=True,
                save_freq='epoch',
            )
        )

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
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.params.epochs,
            callbacks=self.callbacks,
            verbose=max(0, self.params.fit_verbosity),
        )
        if self.params.save_checkpoint and self.params.load_best_checkpoint_after_fit:
            checkpoint = tf.train.Checkpoint(self.model)
            checkpoint.restore(
                str(self.params.checkpoint_path / 'ckpt')).expect_partial()

    def save_model(self, model_path: Path):
        self.model.save(model_path / '1',
                        overwrite=True)
