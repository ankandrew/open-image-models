"""
Public factory for pretrained object detectors.
"""

import os
import warnings
from collections.abc import Sequence
from pathlib import Path

import onnxruntime as ort

from open_image_models.detection.core.base import ObjectDetector
from open_image_models.detection.core.hub import (
    DETECTION_MODELS,
    DetectionModelName,
    DetectorBackend,
    download_model,
)
from open_image_models.detection.core.rf_detr.inference import RFDETRDetector
from open_image_models.detection.core.yolo_v9.inference import YoloV9Detector


def _license_plate_detector(
    detection_model: DetectionModelName,
    conf_thresh: float | None = None,
    providers: Sequence[str | tuple[str, dict]] | None = None,
    sess_options: ort.SessionOptions | None = None,
) -> ObjectDetector:
    """
    Creates a license plate detector using the legacy API.

    Note: This compatibility constructor forwards to `create_detector`. New code should call `create_detector` directly.

    Args:
        detection_model: Name of a registered license plate detection model.
        conf_thresh: Confidence threshold. Uses the model default when omitted.
        providers: ONNX Runtime providers in order of decreasing precedence.
        sess_options: Advanced ONNX Runtime session options.

    Returns:
        A detector configured for the selected license plate model.

    Warns:
        DeprecationWarning: Always emitted because `LicensePlateDetector` is deprecated.
    """
    warnings.warn(
        "LicensePlateDetector is deprecated; use create_detector instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_detector(
        detection_model,
        conf_thresh=conf_thresh,
        providers=providers,
        sess_options=sess_options,
    )


def create_detector(
    model: str | os.PathLike[str],
    *,
    backend: DetectorBackend | None = None,
    class_labels: Sequence[str] | None = None,
    conf_thresh: float | None = None,
    batch_size: int = 1,
    providers: Sequence[str | tuple[str, dict]] | None = None,
    sess_options: ort.SessionOptions | None = None,
) -> ObjectDetector:
    """
    Creates an object detector from a registered model or local ONNX file.

    Args:
        model: Registered model name or path to a local ONNX model.
        backend: Inference backend. Required only for local models.
        class_labels: Model class labels. Required only for local models.
        conf_thresh: Confidence threshold. Uses the model default when omitted.
        batch_size: Maximum inference batch size for models with a dynamic batch dimension.
        providers: ONNX Runtime providers in order of decreasing precedence.
        sess_options: Advanced ONNX Runtime session options.

    Returns:
        A detector configured for the selected model or local file.

    Raises:
        ValueError: If local model metadata is missing or registered model metadata is overridden.
        FileNotFoundError: If a local model file does not exist.
    """
    threshold = conf_thresh
    if not isinstance(model, str) or model not in DETECTION_MODELS:
        model_path = Path(model)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found at '{model_path}'")
        if backend is None or class_labels is None:
            raise ValueError("backend and class_labels are required for a local model")
        if isinstance(class_labels, str) or not class_labels:
            raise ValueError("class_labels must contain at least one label")
        labels = list(class_labels)
    else:
        if backend is not None or class_labels is not None:
            raise ValueError("backend and class_labels cannot override a registered model")
        spec = DETECTION_MODELS[model]
        model_path = download_model(model)
        backend = spec.backend
        labels = list(spec.class_labels)
        if threshold is None:
            threshold = spec.default_conf_thresh

    if backend == "yolo_v9":
        return YoloV9Detector(
            model_path=model_path,
            class_labels=labels,
            conf_thresh=threshold,
            batch_size=batch_size,
            providers=providers,
            sess_options=sess_options,
        )
    if backend == "rf_detr":
        return RFDETRDetector(
            model_path=model_path,
            class_labels=labels,
            conf_thresh=threshold,
            batch_size=batch_size,
            providers=providers,
            sess_options=sess_options,
        )

    raise ValueError(f"Unsupported detector backend: {backend}")
