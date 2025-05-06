# DenseNet on MURA Dataset for Hidden Stratification

This is a PyTorch implementation of a 169-layer [DenseNet](https://arxiv.org/abs/1608.06993) model trained on the MURA dataset, inspired by [arXiv:1712.06957](https://arxiv.org/abs/1712.06957) by Pranav Rajpurkar et al.
It reproduces findings from the paper *Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging* ([arXiv:1909.12475](https://arxiv.org/abs/1909.12475)).

MURA is a large dataset of musculoskeletal radiographs, with each study manually labeled by radiologists as either normal or abnormal. [Learn more](https://stanfordmlgroup.github.io/projects/mura/)

## Setup

- Download the MURA v1.1 image dataset and place it in the root directory of this repo.
- This code is adapted for Mac M4 chips—install a compatible PyTorch version that supports M-series GPUs.
- Create the `./models` and `./checkpoints` directories to store model weights and training checkpoints.
- To evaluate the model, set the `train` flag to `False` in the script to skip training and run evaluation instead.

## Requirements

Install the following dependencies:

- PyTorch
- TorchVision
- NumPy
- Pandas

## Usage

- To train the binary classification model, run:
  ```bash
  python main.py
  ```

- To train the multi-class classification model, run:
  ```bash
  python classification_main.py
  ```
