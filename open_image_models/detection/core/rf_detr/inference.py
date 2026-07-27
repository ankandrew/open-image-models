import logging
import os
import pathlib
from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import onnxruntime as ort

from open_image_models.detection.core.base import (
    ClassLabels,
    DetectionResult,
    draw_detection_results,
    inspect_model_input_shape,
    normalize_class_labels,
    resolve_image_inputs,
)
from open_image_models.detection.core.rf_detr.postprocess import convert_to_detection_result
from open_image_models.detection.core.rf_detr.preprocess import preprocess

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# pylint: disable=duplicate-code


class RFDETRDetector:
    """
    RF-DETR ONNX inference class for exported object detection models.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        class_labels: ClassLabels,
        *,
        conf_thresh: float | None = None,
        num_select: int = 300,
        batch_size: int = 1,
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
    ) -> None:
        """
        Initializes the RF-DETR detector with an exported ONNX model.

        Args:
            model_path: Path to the exported RF-DETR ONNX model.
            class_labels: Contiguous labels or a mapping from class IDs to labels.
            conf_thresh: Confidence threshold for filtering predictions. Defaults to 0.5.
            num_select: Maximum number of query/class pairs to consider.
            batch_size: Maximum inference batch size for dynamic-batch models.
            providers: Optional sequence of providers in order of decreasing precedence. When omitted, CoreML is
                avoided because it may be slower or incompatible with dynamic-batch RF-DETR models.
            sess_options: Advanced session options for ONNX Runtime.
        """
        self.conf_thresh = 0.5 if conf_thresh is None else conf_thresh
        self.class_labels = normalize_class_labels(class_labels)
        self.num_select = num_select
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at '{model_path}'")

        if providers is None:
            providers = [
                provider for provider in ort.get_available_providers() if provider != "CoreMLExecutionProvider"
            ]
        self.model = ort.InferenceSession(str(model_path), providers=providers, sess_options=sess_options)
        self.input_name = self.model.get_inputs()[0].name
        input_shape = inspect_model_input_shape(self.model.get_inputs()[0].shape)
        self.img_size = input_shape.image_size
        self.batch_size = batch_size if input_shape.dynamic_batch else 1
        LOGGER.info("Using ONNX Runtime with %s provider(s)", providers)

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
        Perform object detection on one or multiple images.

        Args:
            images: A single image as a numpy array, a single image path as a string, a list of images as numpy arrays,
                    or a list of image file paths.

        Returns:
            A list of DetectionResult for a single image input,
            or a list of lists of DetectionResult for multiple images.
        """
        loaded_images, is_single = resolve_image_inputs(images)
        results: list[list[DetectionResult]] = []
        for start in range(0, len(loaded_images), self.batch_size):
            results.extend(self._predict_batch(loaded_images[start : start + self.batch_size]))
        return results[0] if is_single else results

    def _predict(self, image: np.ndarray) -> list[DetectionResult]:
        return self._predict_batch([image])[0]

    def _predict_batch(self, images: list[np.ndarray]) -> list[list[DetectionResult]]:
        """
        Performs object detection on an image batch.

        Args:
            images: Input image frames in BGR format.

        Returns:
            Detection results for each image.
        """
        inputs = np.concatenate([preprocess(image, self.img_size) for image in images])
        try:
            boxes, logits = self.model.run(None, {self.input_name: inputs})[:2]
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.warning("An error occurred during model inference: %s", e)
            return [[] for _ in images]

        boxes = np.asarray(boxes)
        logits = np.asarray(logits)
        if boxes.ndim != 3 or logits.ndim != 3 or boxes.shape[0] != len(images) or logits.shape[0] != len(images):
            raise ValueError(f"Expected RF-DETR outputs with batch dimensions, got {boxes.shape} and {logits.shape}")

        return [
            convert_to_detection_result(
                boxes=image_boxes,
                logits=image_logits,
                class_labels=self.class_labels,
                image_size=image.shape[:2],
                score_threshold=self.conf_thresh,
                num_select=self.num_select,
            )
            for image, image_boxes, image_logits in zip(images, boxes, logits, strict=True)
        ]

    def display_predictions(self, image: np.ndarray) -> np.ndarray:
        """
        Run object detection on the input image and display the predictions on the image.

        Args:
            image: An input image as a numpy array.

        Returns:
            The image with bounding boxes and labels drawn on it.
        """
        detections: list[DetectionResult] = self.predict(image)
        return draw_detection_results(image, detections)
