"""
Open Image Models HUB.
"""

import logging
import pathlib
import shutil
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from typing import Literal

from tqdm.asyncio import tqdm

from open_image_models.detection.core.base import ClassLabels
from open_image_models.detection.core.coco import COCO_CLASSES
from open_image_models.utils import safe_write

BASE_URL: str = "https://github.com/ankandrew/open-image-models/releases/download/assets"
"""Base URL where models will be fetched."""
DetectorBackend = Literal["yolo_v9", "rf_detr"]
"""Inference backends supported by the detector factory."""
LicensePlateModelName = Literal[
    "yolo-v9-s-608-license-plate-end2end",
    "yolo-v9-t-640-license-plate-end2end",
    "yolo-v9-t-512-license-plate-end2end",
    "yolo-v9-t-416-license-plate-end2end",
    "yolo-v9-t-384-license-plate-end2end",
    "yolo-v9-t-256-license-plate-end2end",
]
"""Names of the available license plate detection models."""
PlateDetectorModel = LicensePlateModelName
"""Deprecated compatibility alias for `LicensePlateModelName`."""
DetectionModelName = Literal[
    "rf-detr-nano-384-coco",
    "rf-detr-small-512-coco",
    "rf-detr-medium-576-coco",
    "rf-detr-large-704-coco",
    "yolo-v9-s-608-license-plate-end2end",
    "yolo-v9-t-640-license-plate-end2end",
    "yolo-v9-t-512-license-plate-end2end",
    "yolo-v9-t-416-license-plate-end2end",
    "yolo-v9-t-384-license-plate-end2end",
    "yolo-v9-t-256-license-plate-end2end",
]
"""Names of the available object detection models."""

MODEL_CACHE_DIR: pathlib.Path = pathlib.Path.home() / ".cache" / "open-image-models"
"""Default location where models will be stored."""


@dataclass(frozen=True)
class DetectionModelSpec:
    """
    Configuration required to construct a detector for a trained model.

    Attributes:
        url: URL of the ONNX model file.
        backend: Inference backend used by the model.
        class_labels: Labels corresponding to the model's class IDs.
        default_conf_thresh: Default confidence threshold for predictions.
    """

    url: str
    backend: DetectorBackend
    class_labels: ClassLabels
    default_conf_thresh: float


DETECTION_MODELS: dict[DetectionModelName, DetectionModelSpec] = {
    "rf-detr-nano-384-coco": DetectionModelSpec(
        url=f"{BASE_URL}/rf-detr-nano-384-coco.onnx",
        backend="rf_detr",
        class_labels=COCO_CLASSES,
        default_conf_thresh=0.5,
    ),
    "rf-detr-small-512-coco": DetectionModelSpec(
        url=f"{BASE_URL}/rf-detr-small-512-coco.onnx",
        backend="rf_detr",
        class_labels=COCO_CLASSES,
        default_conf_thresh=0.5,
    ),
    "rf-detr-medium-576-coco": DetectionModelSpec(
        url=f"{BASE_URL}/rf-detr-medium-576-coco.onnx",
        backend="rf_detr",
        class_labels=COCO_CLASSES,
        default_conf_thresh=0.5,
    ),
    "rf-detr-large-704-coco": DetectionModelSpec(
        url=f"{BASE_URL}/rf-detr-large-704-coco.onnx",
        backend="rf_detr",
        class_labels=COCO_CLASSES,
        default_conf_thresh=0.5,
    ),
    "yolo-v9-s-608-license-plate-end2end": DetectionModelSpec(
        url=f"{BASE_URL}/yolo-v9-s-608-license-plates-end2end.onnx",
        backend="yolo_v9",
        class_labels=("License Plate",),
        default_conf_thresh=0.25,
    ),
    "yolo-v9-t-640-license-plate-end2end": DetectionModelSpec(
        url=f"{BASE_URL}/yolo-v9-t-640-license-plates-end2end.onnx",
        backend="yolo_v9",
        class_labels=("License Plate",),
        default_conf_thresh=0.25,
    ),
    "yolo-v9-t-512-license-plate-end2end": DetectionModelSpec(
        url=f"{BASE_URL}/yolo-v9-t-512-license-plates-end2end.onnx",
        backend="yolo_v9",
        class_labels=("License Plate",),
        default_conf_thresh=0.25,
    ),
    "yolo-v9-t-416-license-plate-end2end": DetectionModelSpec(
        url=f"{BASE_URL}/yolo-v9-t-416-license-plates-end2end.onnx",
        backend="yolo_v9",
        class_labels=("License Plate",),
        default_conf_thresh=0.25,
    ),
    "yolo-v9-t-384-license-plate-end2end": DetectionModelSpec(
        url=f"{BASE_URL}/yolo-v9-t-384-license-plates-end2end.onnx",
        backend="yolo_v9",
        class_labels=("License Plate",),
        default_conf_thresh=0.25,
    ),
    "yolo-v9-t-256-license-plate-end2end": DetectionModelSpec(
        url=f"{BASE_URL}/yolo-v9-t-256-license-plates-end2end.onnx",
        backend="yolo_v9",
        class_labels=("License Plate",),
        default_conf_thresh=0.25,
    ),
}
"""Detection models available through `create_detector`."""


def _download_with_progress(url: str, filename: pathlib.Path) -> None:
    """
    Downloads a file while displaying progress.

    Args:
        url: URL of the file to download.
        filename: Destination path.
    """
    with urllib.request.urlopen(url) as response, safe_write(filename, mode="wb") as out_file:
        if response.getcode() != HTTPStatus.OK:
            raise ValueError(f"Failed to download file from {url}. Status code: {response.status}")

        file_size = int(response.headers.get("Content-Length", 0))
        desc = f"Downloading {filename.name}"

        with tqdm.wrapattr(out_file, "write", total=file_size, desc=desc) as f_out:
            shutil.copyfileobj(response, f_out)


def download_model(
    model_name: DetectionModelName,
    save_directory: pathlib.Path | None = None,
    force_download: bool = False,
) -> pathlib.Path:
    """
    Download a detection model to a given directory.

    Args:
        model_name: Name of the registered model to download.
        save_directory: Directory in which to store the model.
        force_download: Download the model even if it is already cached.

    Returns:
        Path to the downloaded or cached model.

    Raises:
        ValueError: If the model is unknown or `save_directory` points to a file.
    """
    if model_name not in DETECTION_MODELS:
        available_models = ", ".join(DETECTION_MODELS)
        raise ValueError(f"Unknown model {model_name}. Use one of [{available_models}]")

    if save_directory is None:
        save_directory = MODEL_CACHE_DIR / model_name
    elif save_directory.is_file():
        raise ValueError(f"Expected a directory, but got {save_directory}")

    save_directory.mkdir(parents=True, exist_ok=True)

    model_url = DETECTION_MODELS[model_name].url
    model_filename = save_directory / model_url.split("/")[-1]

    if model_filename.is_file() and not force_download:
        logging.info(
            "Skipping download of '%s' model, already exists at %s",
            model_name,
            save_directory,
        )
        return model_filename

    logging.info("Downloading model to %s", model_filename)
    _download_with_progress(url=model_url, filename=model_filename)

    return model_filename
