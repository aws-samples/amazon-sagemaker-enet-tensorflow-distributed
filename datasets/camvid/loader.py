# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
from pathlib import Path
from PIL import Image
from typing import List, Tuple
import tensorflow as tf
import numpy as np
from math import ceil


class DatasetLoader:
    def __init__(self, base_path: Path, num_shards: int = 1, shard_id: int = 0):
        self.img_width = 960
        self.img_height = 720
        # self.img_width = 960 // 4
        # self.img_height = 720 // 4
        self.train_path = base_path / 'train'
        self.train_labels_path = base_path / 'train_labels'
        self.val_path = base_path / 'val'
        self.val_labels_path = base_path / 'val_labels'
        self.test_path = base_path / 'test'
        self.test_labels_path = base_path / 'test_labels'
        self.report_path = base_path / 'report'

        assert num_shards > 0
        assert shard_id >= 0
        assert shard_id < num_shards
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.load_class_dict()

    def load_class_dict(self):
        with open(self.report_path / 'class_dict.json', 'r') as fp:
            self.class_dict = json.load(fp)

    def list_image_files(self, data_path: Path,
                         labels_path: Path) -> List[Tuple[Path, Path]]:
        image_files = []
        for data_file in data_path.glob('*.png'):
            stem = data_file.stem
            label_file = labels_path / f'{stem}_L.png'
            assert label_file.exists()
            image_files.append((data_file, label_file))

        n = len(image_files)
        shard_size = ceil(float(n) / float(self.num_shards))
        # NB: Make sure that every shard has the same number of items.
        # This is important for data parallel since it expects identical
        # numbers of steps (for oob_allreduce).
        start_idx = (self.shard_id * shard_size) % n
        end_idx = (start_idx + shard_size) % n
        if start_idx < end_idx:
            return image_files[start_idx:end_idx]
        else:
            return image_files[start_idx:] + image_files[:end_idx]

    def _load_image(self, file):
        img = Image.open(file)
        if self.img_width != img.width or self.img_height != img.height:
            img = img.resize((self.img_width, self.img_height),
                             resample=Image.NEAREST)
        img = np.array(img)
        return img

    def load_images(self, image_files):
        for image_file, label_file in image_files:
            img = self._load_image(image_file).astype(np.float32)
            img /= 255.0
            lbl = self._load_image(label_file)
            yield (img, lbl)

    def build_tf_dataset(self, data_path: Path, labels_path: Path) -> tf.data.Dataset:
        image_files = self.list_image_files(data_path, labels_path)

        def images(): return self.load_images(image_files)
        dataset = tf.data.Dataset.from_generator(
            images,
            output_signature=(
                tf.TensorSpec(
                    shape=(self.img_height, self.img_width, 3), dtype=tf.float32),
                tf.TensorSpec(
                    shape=(self.img_height, self.img_width), dtype=tf.uint8),
            )
        )
        return dataset

    @property
    def train(self):
        return self.build_tf_dataset(
            self.train_path,
            self.train_labels_path,
        )

    @property
    def val(self):
        return self.build_tf_dataset(
            self.val_path,
            self.val_labels_path,
        )

    @property
    def test(self):
        return self.build_tf_dataset(
            self.test_path,
            self.test_labels_path,
        )

    @property
    def num_classes(self) -> int:
        return len(self.class_dict)
