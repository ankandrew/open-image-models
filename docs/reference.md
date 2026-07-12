# Object Detection

Use `create_detector` with any registered object detection model. The model registry selects the inference backend,
class labels, and default confidence threshold.

Set `batch_size` above one to batch list inputs when the selected ONNX model has a dynamic batch dimension. Models
exported with a fixed batch size of one automatically retain serial inference.

Local ONNX models require their backend and class labels:

```python
from open_image_models import create_detector

detector = create_detector(
    "/path/to/model.onnx",
    backend="rf_detr",
    class_labels=["vehicle", "License Plate"],
)
```

::: open_image_models.detection.factory.create_detector

# Core API

The `core` module provides base classes and protocols for object detection models, including essential data structures like `BoundingBox` and `DetectionResult`.

### 🔧 Core Components

The following components are shared by all detection backends:

- **`BoundingBox`**: Represents a bounding box for detected objects.
- **`DetectionResult`**: Stores label, confidence, and bounding box for a detection.
- **`ObjectDetector`**: Protocol defining `predict` and `display_predictions`.

::: open_image_models.detection.core.base

::: open_image_models.detection.core.hub
    options:
      group_by_category: false
      members:
        - DetectionModelName
        - DetectionModelSpec
        - DETECTION_MODELS
