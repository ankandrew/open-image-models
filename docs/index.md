# Open Image Models

**Ready-to-use** models for a range of **computer vision** tasks like **detection**, **classification**, and
**more**. With **ONNX** support, you get **fast** and **accurate** results right out of the box.

Easily integrate these models into your apps for **real-time** processing—ideal for edge devices, cloud setups, or
production environments. In **one line of code**, you can have **powerful** model **inference** running!

```python
from open_image_models import LicensePlateDetector

lp_detector = LicensePlateDetector(detection_model="yolo-v9-t-256-license-plate-end2end")
lp_detector.predict("path/to/license_plate_image.jpg")
```

???+ info
    ✨ That’s it! Fast, accurate computer vision models with just one line of code.

## Features

- 🚀 Pre-trained: Models are **ready** for immediate use, no additional training required.
- 🌟 ONNX: Cross-platform support for **fast inference** on both CPU and GPU environments.
- ⚡ Performance: Optimized for both speed and accuracy, ensuring efficient **real-time** applications.
- 💻 Simple API: Power up your applications with robust model inference in just one line of code.
