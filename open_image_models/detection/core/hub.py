"""
Open Image Models HUB.
"""

import logging
import pathlib
import shutil
import urllib.request
from http import HTTPStatus
from typing import Literal

from tqdm.asyncio import tqdm

from open_image_models.utils import safe_write

BASE_URL: str = "https://github.com/ankandrew/open-image-models/releases/download"
"""Base URL where models will be fetched."""
PlateDetectorModel = Literal[
    "yolo-v9-s-608-license-plate-end2end",
    "yolo-v9-t-640-license-plate-end2end",
    "yolo-v9-t-512-license-plate-end2end",
    "yolo-v9-t-416-license-plate-end2end",
    "yolo-v9-t-384-license-plate-end2end",
    "yolo-v9-t-256-license-plate-end2end",
]
"""Available ONNX models for doing detection."""

AVAILABLE_ONNX_MODELS: dict[PlateDetectorModel, str] = {
    # Plate Detection
    "yolo-v9-s-608-license-plate-end2end": f"{BASE_URL}/assets/yolo-v9-s-608-license-plates-end2end.onnx",
    "yolo-v9-t-640-license-plate-end2end": f"{BASE_URL}/assets/yolo-v9-t-640-license-plates-end2end.onnx",
    "yolo-v9-t-512-license-plate-end2end": f"{BASE_URL}/assets/yolo-v9-t-512-license-plates-end2end.onnx",
    "yolo-v9-t-416-license-plate-end2end": f"{BASE_URL}/assets/yolo-v9-t-416-license-plates-end2end.onnx",
    "yolo-v9-t-384-license-plate-end2end": f"{BASE_URL}/assets/yolo-v9-t-384-license-plates-end2end.onnx",
    "yolo-v9-t-256-license-plate-end2end": f"{BASE_URL}/assets/yolo-v9-t-256-license-plates-end2end.onnx",
}
"""Available ONNX models for doing inference."""
MODEL_CACHE_DIR: pathlib.Path = pathlib.Path.home() / ".cache" / "open-image-models"
"""Default location where models will be stored."""


def _download_with_progress(url: str, filename: pathlib.Path) -> None:
    """
    Download utility function with progress bar.

    :param url: URL of the model to download.
    :param filename: Where to save the model.
    """
    with urllib.request.urlopen(url) as response, safe_write(filename, mode="wb") as out_file:
        if response.getcode() != HTTPStatus.OK:
            raise ValueError(f"Failed to download file from {url}. Status code: {response.status}")

        file_size = int(response.headers.get("Content-Length", 0))
        desc = f"Downloading {filename.name}"

        with tqdm.wrapattr(out_file, "write", total=file_size, desc=desc) as f_out:
            shutil.copyfileobj(response, f_out)


def download_model(
    model_name: PlateDetectorModel,
    save_directory: pathlib.Path | None = None,
    force_download: bool = False,
) -> pathlib.Path:
    """
    Download a detection model to a given directory.

    :param model_name: Which model to download.
    :param save_directory: Directory to save the model. It should point to a folder. If not supplied, this will point
    to '~/.cache/<model_name>'
    :param force_download: Force and download the model if it already exists in `save_directory`.
    :return: Path where the model lives.
    """
    if model_name not in AVAILABLE_ONNX_MODELS:
        available_models = ", ".join(AVAILABLE_ONNX_MODELS.keys())
        raise ValueError(f"Unknown model {model_name}. Use one of [{available_models}]")

    if save_directory is None:
        save_directory = MODEL_CACHE_DIR / model_name
    elif save_directory.is_file():
        raise ValueError(f"Expected a directory, but got {save_directory}")

    save_directory.mkdir(parents=True, exist_ok=True)

    model_url = AVAILABLE_ONNX_MODELS[model_name]
    model_filename = save_directory / model_url.split("/")[-1]

    if not force_download and model_filename.is_file():
        logging.info(
            "Skipping download of '%s' model, already exists at %s",
            model_name,
            save_directory,
        )
        return model_filename

    # Download the model if not present or if we want to force the download
    if force_download or not model_filename.is_file():
        logging.info("Downloading model to %s", model_filename)
        _download_with_progress(url=model_url, filename=model_filename)

    return model_filename
