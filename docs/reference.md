# 🛠 Pipelines Overview

## License Plate Detection

🚗 **License Plate Detection** allows you to detect and identify license plates in images using a specialized pipeline based on the YOLOv9 model.

The `LicensePlateDetector` is specialized for license plate detection. It utilizes the **YOLOv9** object detection model to recognize license plates in images.

::: open_image_models.detection.pipeline.license_plate.LicensePlateDetector

---

# Core API Documentation

The `core` module provides base classes and protocols for object detection models, including essential data structures like `BoundingBox` and `DetectionResult`.

### 🔧 Core Components

The following components are used across detection pipelines and models:

- **`BoundingBox`**: Represents a bounding box for detected objects.
- **`DetectionResult`**: Stores label, confidence, and bounding box for a detection.
- **`ObjectDetector`**: Protocol defining essential methods like `predict`, `show_benchmark`, and `display_predictions`.

??? abstract "Core API Documentation"
    ::: open_image_models.detection.core.base
