<p>
  <img src="./assets/open-image-models-logo.png" alt="Open Image Models Logo" width="400"/>
</p>

# Fast & Lightweight License Plate OCR

[![Actions status](https://github.com/ankandrew/fast-plate-ocr/actions/workflows/main.yaml/badge.svg)](https://github.com/ankandrew/fast-plate-ocr/actions)
[![Keras 3](https://img.shields.io/badge/Keras-3-red?logo=keras&logoColor=red&labelColor=white)](https://keras.io/keras_3/)
[![image](https://img.shields.io/pypi/v/fast-plate-ocr.svg)](https://pypi.python.org/pypi/fast-plate-ocr)
[![image](https://img.shields.io/pypi/pyversions/fast-plate-ocr.svg)](https://pypi.python.org/pypi/fast-plate-ocr)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![image](https://img.shields.io/pypi/l/fast-plate-ocr.svg)](https://pypi.python.org/pypi/fast-plate-ocr)

<!-- TOC -->
* [Fast & Lightweight License Plate OCR](#fast--lightweight-license-plate-ocr)
  * [Introduction](#introduction)
  * [Features](#features)
  * [Available Models](#available-models)
    * [Object Detection](#object-detection)
      * [Plate Detection](#plate-detection)
    * [Contributing](#contributing)
<!-- TOC -->

---

## Introduction

We aim to provide a variety of **pre-trained** models for different **computer vision** tasks, such as object detection and
image classification, that can be used **out-of-the-box** for **fast** inference using ONNX. These models are optimized for
performance and accuracy across various image sizes and tasks.

Models found here can easily be integrated into your applications for real-time processing, making them ideal for
deployment in edge devices, cloud environments, or production systems.

## Features

- 🚀 Pre-trained Models: Models are **ready** for immediate use, no additional training required.
- 🔄 ONNX Format: Cross-platform support for **fast inference** on both CPU and GPU environments.
- ⚡ High Performance: Optimized for both speed and accuracy, ensuring efficient **real-time** applications.
- 📏 Variety of Image Sizes: Models **available** with different input sizes, allowing flexibility based on the task's
  performance and speed requirements.
- 📊 Evaluation Metrics: Precision, Recall, mAP50, and mAP50-95 metrics are provided for each model, helping users select
  the most appropriate model for their needs.

## Available Models

### Object Detection

#### Plate Detection

| Model    | Image Size | Precision (P) | Recall (R) | mAP50 | mAP50-95 | Speed (ms) |
|----------|------------|---------------|------------|-------|----------|------------|
| yolov9-t | 640        | 0.955         | 0.91       | 0.959 | 0.75     | XXX        |
| yolov9-t | 512        | 0.948         | 0.901      | 0.95  | 0.718    | XXX        |
| yolov9-t | 384        | 0.943         | 0.863      | 0.921 | 0.688    | XXX        |
| yolov9-t | 256        | 0.937         | 0.797      | 0.858 | 0.606    | XXX        |

_<sup>[1]</sup> Inference on Mac M1 chip using CPUExecutionProvider. Utilizing CoreMLExecutionProvider accelerates speed
by 5x._

### Contributing

Contributions to the repo are greatly appreciated. Whether it's bug fixes, feature enhancements, or new models,
your contributions are warmly welcomed.

To start contributing or to begin development, you can follow these steps:

1. Clone repo
    ```shell
    git clone https://github.com/ankandrew/fast-plate-ocr.git
    ```
2. Install all dependencies using [Poetry](https://python-poetry.org/docs/#installation):
    ```shell
    poetry install --all-extras
    ```
3. To ensure your changes pass linting and tests before submitting a PR:
    ```shell
    make checks
    ```
