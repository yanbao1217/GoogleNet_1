# GoogleNet for FashionMNIST Classification

This repository contains an implementation of **GoogleNet** with custom **Inception modules** in **PyTorch**, applied to the **FashionMNIST** dataset. The project supports training, validation, and visualization of loss and accuracy curves.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation & Visualization](#evaluation--visualization)
- [Model Architecture](#model-architecture)

---

## Project Overview

This project implements:

- **GoogleNet** (Inception v1) for image classification
- Custom **Inception module**
- Training and validation pipeline with **PyTorch**
- Visualization of **loss** and **accuracy** curves per epoch
- FashionMNIST dataset preprocessing and resizing to **224x224** for GoogleNet input

---

## Requirements

- Python 3.8+
- PyTorch 1.10.1
- torchvision
- numpy
- pandas
- matplotlib

Install requirements via pip:

```bash
pip install torch torchvision numpy pandas matplotlib
```
or via conda:
```bash
conda install pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch
conda install numpy pandas matplotlib
```
---
## Dataset

This project uses FashionMNIST:
- Training set: 60,000 images
- Test set: 10,000 images
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Images are automatically resized to 224x224 to match GoogleNet input requirements.

---
## Installation

Clone this repository:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

---
## Usage

###  Run the training and evaluation script:
```bash
python train.py
```

### The script automatically:
- Downloads FashionMNIST dataset
- Splits training/validation (80/20)
- Trains the GoogleNet model for 20 epochs
- Saves the best model weights as best_model.pth
- Plots loss and accuracy curves

---
## Training
- Optimizer: Adam, learning rate = 0.001
- Loss function: CrossEntropyLoss
- Batch size: 32
- Number of epochs: configurable (default: 20)
- Device: automatically uses GPU if available, otherwise CPU

---
## Evaluation & Visualization
- The training script tracks:
- Training loss & accuracy
- Validation loss & accuracy
- After training, a matplotlib figure is generated showing:
- Loss curves (train & validation)
- Accuracy curves (train & validation)