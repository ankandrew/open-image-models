"""
License Plate object detection pipeline.
"""

import logging
import os
from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import onnxruntime as ort

from open_image_models.detection.core.base import DetectionResult
from open_image_models.detection.core.hub import PlateDetectorModel, download_model
from open_image_models.detection.core.yolo_v9.inference import YoloV9ObjectDetector

# Setup logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class LicensePlateDetector(YoloV9ObjectDetector):
    """
    Specialized detector for license plates using YoloV9 model.
    Inherits from YoloV9ObjectDetector and sets up license plate specific configuration.
    """

    def __init__(
        self,
        detection_model: PlateDetectorModel,
        conf_thresh: float = 0.25,
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions = None,
    ) -> None:
        """
        Initializes the LicensePlateDetector with the specified detection model and inference device.

        Args:
            detection_model: Detection model to use, see `PlateDetectorModel`.
            conf_thresh: Confidence threshold for filtering predictions.
            providers: Optional sequence of providers in order of decreasing precedence. If not specified, all available
                providers are used.
            sess_options: Advanced session options for ONNX Runtime.
        """
        # Download model if needed
        detector_model_path = download_model(detection_model)
        super().__init__(
            model_path=detector_model_path,
            conf_thresh=conf_thresh,
            class_labels=["License Plate"],
            providers=providers,
            sess_options=sess_options,
        )
        LOGGER.info("Initialized LicensePlateDetector with model %s", detector_model_path)

    # pylint: disable=duplicate-code
    @overload
    def predict(self, images: np.ndarray) -> list[DetectionResult]: ...

    @overload
    def predict(self, images: list[np.ndarray]) -> list[list[DetectionResult]]: ...

    @overload
    def predict(self, images: str) -> list[DetectionResult]: ...

    @overload
    def predict(self, images: list[str]) -> list[list[DetectionResult]]: ...

    @overload
    def predict(self, images: os.PathLike[str]) -> list[DetectionResult]: ...

    @overload
    def predict(self, images: list[os.PathLike[str]]) -> list[list[DetectionResult]]: ...

    def predict(self, images: Any) -> list[DetectionResult] | list[list[DetectionResult]]:
        """
        Perform license plate detection on one or multiple images.

        This method is a specialized version of the `YoloV9ObjectDetector.predict` method,
        focusing on detecting license plates in images.

        Args:
            images: A single image as a numpy array, a single image path as a string, a list of images as numpy arrays,
                    or a list of image file paths.

        Returns:
            A list of `DetectionResult` for a single image input, or a list of lists of `DetectionResult` for multiple
                images.

        Example usage:

        ```python
        from open_image_models import LicensePlateDetector

        lp_detector = LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")
        lp_detector.predict("path/to/license_plate_image.jpg")
        ```

        Raises:
            ValueError: If the image could not be loaded or processed.
        """
        return super().predict(images)
