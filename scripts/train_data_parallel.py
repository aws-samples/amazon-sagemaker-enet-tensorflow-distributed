# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
from pathlib import Path
import os

import tensorflow as tf
import smdistributed.dataparallel.tensorflow as sdp

from datasets.camvid.loader import DatasetLoader
from enet.model import ENetParams, ENetModel
from trainer.logging import SMMetricsLogger
from trainer.data_parallel import TrainerParams, Trainer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def setup_smdistributed():
    sdp.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(
            gpus[sdp.local_rank()], 'GPU')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dropout-rate1', type=float, default=0.01)
    parser.add_argument('--dropout-rate2', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--data-path', type=str, default='/opt/ml/input/data')
    parser.add_argument('--checkpoint-path', type=str,
                        default='/opt/ml/checkpoints')
    parser.add_argument('--model-path', type=str, default='/opt/ml/model')

    args, _ = parser.parse_known_args()
    return args


def build_trainer(args) -> Trainer:
    loader = DatasetLoader(Path(args.data_path).expanduser(),
                           num_shards=sdp.size(), shard_id=sdp.rank())

    enet_params = ENetParams(
        input_dim=(loader.img_height, loader.img_width, 3),
        num_object_classes=loader.num_classes,
        dropout_rate1=args.dropout_rate1,
        dropout_rate2=args.dropout_rate2,
    )
    model = ENetModel(enet_params)

    trainer_params = TrainerParams(
        num_classes=loader.num_classes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        save_checkpoint=True,
        checkpoint_path=Path(args.checkpoint_path).expanduser(),
        load_best_checkpoint_after_fit=True,
    )
    metrics_logger = SMMetricsLogger()
    trainer = Trainer(trainer_params, loader, model,
                      extra_callbacks=[metrics_logger])

    return trainer


def train(args, trainer: Trainer):
    trainer.fit()
    trainer.save_model(Path(args.model_path).expanduser())


if __name__ == '__main__':
    setup_smdistributed()
    args = parse_args()
    trainer = build_trainer(args)
    train(args, trainer)
