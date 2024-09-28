"""
FIXME
"""

import logging
import os
import pathlib
from collections.abc import Sequence

import numpy as np
import onnxruntime as ort

from open_image_models.detection.core.base import DetectionResult, ObjectDetector
from open_image_models.detection.core.yolo_v9.postprocess import convert_to_detection_result
from open_image_models.detection.core.yolo_v9.preprocess import preprocess
from open_image_models.utils import log_time_taken

# Setup logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)  # Set the logging level accordingly (DEBUG, INFO, etc.)


class YoloV9ObjectDetector(ObjectDetector):
    """
    YoloV9-specific ONNX inference class for performing object detection using the Yolo v9 ONNX model.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        class_labels: list[str],
        conf_thresh: float = 0.25,
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions = None,
    ) -> None:
        """
        Initializes the YoloV9ObjectDetector with the specified detection model and inference device.

        Args:
            model_path: Path to the ONNX model file to use.
            class_labels: List of class labels corresponding to the class IDs.
            conf_thresh: Confidence threshold for filtering predictions.
            providers: Optional sequence of providers in order of decreasing precedence. If not specified, all available
            providers are used.
            sess_options: Advanced session options for ONNX Runtime.
        """
        self.conf_thresh = conf_thresh
        self.class_labels = class_labels

        # Check if model path exists
        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at '{model_path}'")

        providers = providers or ort.get_available_providers()
        self.model = ort.InferenceSession(str(model_path), providers=providers, sess_options=sess_options)

        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        _, _, h, w = self.model.get_inputs()[0].shape
        assert h == w, f"{h} and {w} don't match, squared images only supported"
        self.img_size = h, w

        self.providers = providers
        LOGGER.info("Using ONNX Runtime with %s provider(s)", self.providers)

    def predict(self, image: np.ndarray) -> list[DetectionResult]:
        """
        Perform object detection on a single image frame.

        This function takes an image in BGR format, runs it through the object detection model,
        and returns a list of detected objects, including their class labels and bounding boxes.

        Args:
            image: Input image frame in BGR format.

        Returns:
            A list of DetectionResult containing detected objects information.
        """
        # Preprocess the image using YoloV9-specific preprocessing function
        inputs, ratio, (dw, dh) = preprocess(image, self.img_size)
        # Run inference
        with log_time_taken("ONNX end2end inference"):
            predictions = self.model.run([self.output_name], {self.input_name: inputs})[0]
        # Convert raw predictions to a list of DetectionResult objects
        return convert_to_detection_result(predictions, self.class_labels, ratio, (dw, dh))
