# Open Image Models

[![Actions status](https://github.com/ankandrew/open-image-models/actions/workflows/main.yaml/badge.svg)](https://github.com/ankandrew/open-image-models/actions)
[![image](https://img.shields.io/pypi/v/open-image-models.svg)](https://pypi.python.org/pypi/open-image-models)
[![image](https://img.shields.io/pypi/pyversions/open-image-models.svg)](https://pypi.python.org/pypi/open-image-models)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![image](https://img.shields.io/pypi/l/open-image-models.svg)](https://pypi.python.org/pypi/open-image-models)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces/ankandrew/open-image-models)

<p>
  <img src="https://raw.githubusercontent.com/ankandrew/open-image-models/4053f058a80e5ba7edf5d59614e9f8167f83c0f2/assets/open-image-models-logo.png" alt="Open Image Models Logo" width="650"/>
</p>

<!-- TOC -->
* [Open Image Models](#open-image-models)
  * [Introduction](#introduction)
  * [Features](#features)
  * [Available Models](#available-models)
    * [Object Detection](#object-detection)
      * [Plate Detection](#plate-detection)
    * [Installation](#installation)
    * [Contributing](#contributing)
<!-- TOC -->

---

## Introduction

**Ready-to-use** models for a range of **computer vision** tasks like **detection**, **classification**, and
**more**. With **ONNX** support, you get **fast** and **accurate** results right out of the box.

Easily integrate these models into your apps for **real-time** processing—ideal for edge devices, cloud setups, or
production environments. In **one line of code**, you can have **powerful** model **inference** running!

```python
from open_image_models import LicensePlateDetector

lp_detector = LicensePlateDetector(detection_model="yolo-v9-t-256-license-plate-end2end")
lp_detector.predict("path/to/license_plate_image.jpg")
```

✨ That's it! Powerful license plate detection with just a few lines of code.

## Features

- 🚀 Pre-trained: Models are **ready** for immediate use, no additional training required.
- 🌟 ONNX: Cross-platform support for **fast inference** on both CPU and GPU environments.
- ⚡ Performance: Optimized for both speed and accuracy, ensuring efficient **real-time** applications.
- 💻 Simple API: Power up your applications with robust model inference in just one line of code.

## Available Models

### Object Detection

#### Plate Detection

![](https://raw.githubusercontent.com/ankandrew/LocalizadorPatentes/2e765012f69c4fbd8decf998e61ed136004ced24/extra/demo_localizador.gif)

| Model    | Image Size | Precision (P) | Recall (R) | mAP50 | mAP50-95 | Speed (ms) <sup>[1]</sup> |
|----------|------------|---------------|------------|-------|----------|---------------------------|
| yolov9-t | 640        | 0.955         | 0.91       | 0.959 | 0.75     | 9.96                      |
| yolov9-t | 512        | 0.948         | 0.901      | 0.95  | 0.718    | 7.10                      |
| yolov9-t | 384        | 0.943         | 0.863      | 0.921 | 0.688    | 5.76                      |
| yolov9-t | 256        | 0.937         | 0.797      | 0.858 | 0.606    | 2.65                      |

_<sup>[1]</sup> Inference on a Mac M1._

<details>
  <summary>Usage</summary>

  ```python
import cv2
from rich import print

from open_image_models import LicensePlateDetector

# Initialize the License Plate Detector with the pre-trained YOLOv9 model
lp_detector = LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

# Load an image
image_path = "path/to/license_plate_image.jpg"
image = cv2.imread(image_path)

# Perform license plate detection
detections = lp_detector.predict(image)
print(detections)

# Benchmark the model performance
lp_detector.show_benchmark(num_runs=1000)

# Display predictions on the image
annotated_image = lp_detector.display_predictions(image)

# Show the annotated image
cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
  ```

</details>

### Installation

To install open-image-models via pip, use the following command:

```shell
pip install open-image-models
```

### Contributing

Contributions to the repo are greatly appreciated. Whether it's bug fixes, feature enhancements, or new models,
your contributions are warmly welcomed.

To start contributing or to begin development, you can follow these steps:

1. Clone repo
    ```shell
    git clone https://github.com/ankandrew/open-image-models.git
    ```
2. Install all dependencies using [Poetry](https://python-poetry.org/docs/#installation):
    ```shell
    poetry install --all-extras
    ```
3. To ensure your changes pass linting and tests before submitting a PR:
    ```shell
    make checks
    ```
