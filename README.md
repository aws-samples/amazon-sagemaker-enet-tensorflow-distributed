# ENet Model with Distributed Training on Amazon SageMaker

This repository contains a TensorFlow 2 implementation of the [ENet](https://arxiv.org/abs/1606.02147) model
with support for distributed training on Amazon SageMaker.

Compared to [other examples](https://sagemaker-examples.readthedocs.io/en/latest/training/distributed_training/index.html) 
that use the SageMaker distributed training libraries, the examples in this repository 
strike a balance between functionality and complexity.
The training code is relatively straightforward and easy to understand.
However it offers more features that are beyond basic starter examples:
* Model checkpointing and restoring
* Training metrics (in addition to loss)
* Support for a validation dataset
* Logging of training / validation loss & metrics for CloudWatch integration

## Overview

### ENet Model

Abstract from the [Paper](https://arxiv.org/abs/1606.02147):
> The ability to perform pixel-wise semantic segmentation in real-time is of paramount importance in mobile applications. Recent deep neural networks aimed at this task have the disadvantage of requiring a large number of floating point operations and have long run-times that hinder their usability. In this paper, we propose a novel deep neural network architecture named ENet (efficient neural network), created specifically for tasks requiring low latency operation. ENet is up to 18× faster, requires 75× less FLOPs, has 79× less parameters, and provides similar or better accuracy to existing models. We have tested it on CamVid, Cityscapes and SUN datasets and report on comparisons with existing state-of-the-art methods, and the trade-offs between accuracy and processing time of a network. We present performance measurements of the proposed architecture on embedded systems and suggest possible software improvements that could make ENet even faster.

ENet is a deep learning model for semantic image segmentation which
is a resource intensive task resulting in large models with millions of parameters.

For such models distributed training helps to speed up model development.
This sample repository demonstrates how ENet can be trained using various
parallelisation techniques available in [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

### Distributed Training Techniques

SageMaker provides distributed training libraries for [data parallelism](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html) and [model parallelism](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html).
The libraries are optimized for the SageMaker training environment, help adapt your distributed training jobs to SageMaker, and improve training speed and throughput.

With only a few lines of code, you can add either data parallelism or model parallelism to your TensorFlow or PyTorch training scripts. Model parallelism splits models too large to fit on a single GPU into smaller parts before distributing across multiple GPUs to train, and data parallelism splits large datasets to train concurrently in order to improve training speed.

In computer vision, hardware constraints often force data scientists to pick batch sizes or input sizes that are smaller than they would prefer. For example, bigger inputs may improve model accuracy but may cause out-of-memory errors and poor performance with smaller batch sizes. Similarly, larger batch sizes improve GPU utilization and performance but may hinder model accuracy. SageMaker distribute training libraries offer the flexibility to easily train models efficiently with lower batch sizes or train with bigger inputs.

## Code Organisation

This repository contains an implementation of ENet using TensorFlow 2.x,
along with training scripts for single and multiple (parallel) GPUs and instances.
It uses the [CamVid (Cambridge-driving Labeled Video Database)](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
dataset for training the model.

The source code is organised in the following modules:
* `enet`: Implementation of the ENet model in TensorFlow 2.x
  * `model`: Implementation of the model
  * `layers`: Implementation of the network's layers
  * `utils`: Code helpers and utilities
* `trainer`: Training mode implementations
  * `single`: Training on a single device
  * `data_parallel`: Training routine for data parallel training
  * `model_parallel`: Training routine for model parallel training
  * `logging`: TensorFlow callback for logging training metrics that can be parsed by Amazon SageMaker
  * `metrics`: Custom implementation of a multi-class IoU metric for TensorFlow
* `datasets.camvid`: Data loader and preprocessing logic for the CamVid dataset

To interact with the implementation, the following scripts and notebooks are provided:
* `scripts/train_single.py`: SageMaker training script for a single device
* `scripts/train_data_parallel.py`: SageMaker data parallel training script
* `scripts/train_model_parallel.py`: SageMaker model parallel training script
* `notebooks/preprocess-camvid.ipynb`: Notebook for preparing the CamVid dataset
* `notebooks/train-single.ipynb`: Example notebook that trains ENet on a single GPU
* `notebooks/train-dataparallel.ipynb`: ENet with data parallel training
* `notebooks/train-modelparallel.ipynb`: ENet with model parallel training

## Setup Instructions on Amazon SageMaker

1. If you don't already have a SageMaker Studio domain and user, create them by following [this](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks.html) guide in the documentation.
2. Clone this repository in your SageMaker Studio Lab environment ([guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-use-external.html#studio-lab-use-external-clone-github)).
3. Download the CamVid dataset from Kaggle and upload it to S3 (as described in the [preprocess-camvid.ipynb](./notebooks/preprocess-camvid.ipynb) notebook).
4. Step through the [preprocess-camvid.ipynb](./notebooks/preprocess-camvid.ipynb) notebook to create a preprocessed dataset.
5. Look at the `train-*.ipynb` notebooks and the referenced training scripts for examples of distributed model training in Amazon SageMaker.