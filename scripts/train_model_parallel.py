# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
from pathlib import Path
import os

import tensorflow as tf
import smdistributed.modelparallel.tensorflow as smp

from datasets.camvid.loader import DatasetLoader
from enet.model import ENetParams
from trainer.logging import SMMetricsLogger
from trainer.model_parallel import DistributedENet, TrainerParams, Trainer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def setup_smdistributed():
    smp.init()


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
                           num_shards=1, shard_id=0)

    enet_params = ENetParams(
        input_dim=(loader.img_height, loader.img_width, 3),
        num_object_classes=loader.num_classes,
        dropout_rate1=args.dropout_rate1,
        dropout_rate2=args.dropout_rate2,
    )
    model = DistributedENet(enet_params)

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
                      input_dim=enet_params.input_dim,
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
