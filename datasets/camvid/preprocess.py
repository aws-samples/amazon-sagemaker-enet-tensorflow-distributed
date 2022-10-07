# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Dict, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-path', type=str,
                        default='/opt/ml/processing/input')
    parser.add_argument('--output-path', type=str,
                        default='/opt/ml/processing/output')
    parser.add_argument('--report-path', type=str,
                        default='/opt/ml/processing/report')

    args, _ = parser.parse_known_args()
    return args


def copy_images(source_path: Path, destination_path: Path):
    destination_path.mkdir(parents=True, exist_ok=True)
    for src in tqdm(source_path.glob('*.png')):
        # src.rename(destination_path / src.name)
        shutil.copyfile(src, destination_path / src.name)


ClassDict = Dict[Tuple[int, int, int], Tuple[int, str]]


def load_class_dict(path: Path) -> ClassDict:
    class_dict = {}
    with open(path, 'r') as fp:
        reader = csv.reader(fp)
        next(reader)
        for i, row in enumerate(reader):
            name, r, g, b = tuple(row)
            key = (int(r), int(g), int(b))
            assert key not in class_dict
            class_dict[key] = (i, name)
    return class_dict


def _extract_class_value_for_name(name: str, class_dict: ClassDict) -> np.uint8:
    for v in class_dict.values():
        if v[1] == name:
            return np.uint8(v[0])
    raise ValueError(f'Name {name} not found in class dict')


def transform_label_image(img: Image, class_dict: ClassDict) -> Image:
    void_val = _extract_class_value_for_name('Void', class_dict)
    imgarr = np.array(img)
    transformed = np.full(shape=imgarr.shape[:2],
                          fill_value=void_val,
                          dtype=np.uint8)
    for (r, g, b), (val, _) in class_dict.items():
        const_val = np.zeros_like(imgarr, dtype=np.uint8)
        const_val[:, :, 0] = np.uint8(r)
        const_val[:, :, 1] = np.uint8(g)
        const_val[:, :, 2] = np.uint8(b)
        lbl_mask = np.all(imgarr == const_val, axis=-1)
        transformed[lbl_mask] = np.uint8(val)
    transformed = Image.fromarray(transformed, mode='L')
    return transformed


def preprocess_labels(source_path: Path, destination_path: Path,
                      class_dict: ClassDict):
    destination_path.mkdir(parents=True, exist_ok=True)
    cnt = 0
    for src in tqdm(source_path.glob('*.png')):
        src_img = Image.open(src)
        dst_img = transform_label_image(src_img, class_dict)
        dst_img.save(destination_path / src.name)
        cnt += 1
    return cnt


def write_preprocessing_report(report_path: Path,
                               num_train: int,
                               num_val: int,
                               num_test: int):
    preprocessing_report = {
        'preprocessing': {
            'dataset': {
                'num_train_samples': num_train,
                'num_val_samples': num_val,
                'num_test_samples': num_test,
            }
        }
    }
    json.dump(preprocessing_report,
              open(report_path / 'preprocessing_report.json', 'w'),
              indent=2)


def write_class_dict(report_path: Path,
                     class_dict: ClassDict):
    cd = {v[0]: (v[1], k) for k, v in class_dict.items()}
    json.dump(cd,
              open(report_path / 'class_dict.json', 'w'),
              indent=2)


def preprocess():
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.mkdir(parents=True, exist_ok=True)

    assert (input_path / 'train').exists()
    assert (input_path / 'train_labels').exists()
    assert (input_path / 'val').exists()
    assert (input_path / 'val_labels').exists()
    assert (input_path / 'test').exists()
    assert (input_path / 'test_labels').exists()
    assert (input_path / 'class_dict.csv').exists()

    print('Copying training images')
    copy_images(input_path / 'train', output_path / 'train')
    print('Copying validation images')
    copy_images(input_path / 'val', output_path / 'val')
    print('Copying test images')
    copy_images(input_path / 'test', output_path / 'test')
    class_dict = load_class_dict(input_path / 'class_dict.csv')
    print('Preprocessing training labels')
    num_train = preprocess_labels(input_path / 'train_labels',
                                  output_path / 'train_labels',
                                  class_dict)
    print('Preprocessing validation labels')
    num_val = preprocess_labels(input_path / 'val_labels',
                                output_path / 'val_labels',
                                class_dict)
    print('Preprocessing test labels')
    num_test = preprocess_labels(input_path / 'test_labels',
                                 output_path / 'test_labels',
                                 class_dict)

    write_preprocessing_report(report_path,
                               num_train, num_val, num_test)
    write_class_dict(report_path, class_dict)


if __name__ == '__main__':
    preprocess()
