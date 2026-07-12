"""
Open Image Models package.
"""

import warnings

try:
    from open_image_models.detection.core.base import BoundingBox, DetectionResult, ObjectDetector
    from open_image_models.detection.factory import _license_plate_detector, create_detector
except ModuleNotFoundError as error:
    if error.name != "onnxruntime":
        raise
    warnings.warn(
        "[open_image_models] create_detector is unavailable. "
        "Install an ONNX Runtime extra, such as `open-image-models[onnx]`.",
        stacklevel=2,
    )
    __all__ = ["BoundingBox", "DetectionResult", "ObjectDetector"]
else:
    LicensePlateDetector = _license_plate_detector  # pylint: disable=invalid-name
    __all__ = ["BoundingBox", "DetectionResult", "LicensePlateDetector", "ObjectDetector", "create_detector"]
