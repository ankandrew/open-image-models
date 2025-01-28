import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import no_type_check

import cv2
import numpy as np
import pytest

from open_image_models.detection.core.base import DetectionResult
from open_image_models.detection.core.hub import download_model
from open_image_models.detection.core.yolo_v9.inference import YoloV9ObjectDetector
from test.assets import ASSETS_DIR


@pytest.fixture(name="model_path", scope="module")
def _model_path() -> Iterator[Path]:
    """
    Fixture for downloading and providing a valid ONNX model path using the `download_model` function.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Use real detector, can be any (i.e. plate detector)
        model_path = download_model(
            model_name="yolo-v9-t-384-license-plate-end2end",
            save_directory=Path(tmp_dir),
            force_download=False,
        )
        # Yield the path to the model in the temporary directory
        yield model_path


@pytest.fixture(name="mock_class_labels", scope="module")
def _mock_class_labels() -> list[str]:
    """Fixture for class labels."""
    return ["License Plate"]


@pytest.fixture(name="mock_image", scope="module")
def _mock_image() -> np.ndarray:
    """Fixture to provide a sample image loaded from a real image file (BGR format)."""
    # Define the path to the real image
    image_path = ASSETS_DIR / "car_image.webp"
    # Load the image using OpenCV
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image


@pytest.fixture(name="mock_image_path", scope="module")
def _mock_image_path() -> Iterator[Path]:
    """Fixture to create and return a valid image file path."""
    yield ASSETS_DIR / "car_image.webp"


@pytest.fixture(name="yolo_detector", scope="module")
def _yolo_detector(model_path: Path, mock_class_labels: list[str]) -> YoloV9ObjectDetector:
    """Fixture to initialize the YoloV9ObjectDetector with a mock model path and class labels."""
    return YoloV9ObjectDetector(
        model_path=model_path, class_labels=mock_class_labels, conf_thresh=0.25, providers=["CPUExecutionProvider"]
    )


def test_predict_with_image_array(yolo_detector: YoloV9ObjectDetector, mock_image: np.ndarray) -> None:
    """Test the predict method with a single image array."""
    result: list[DetectionResult] = yolo_detector.predict(mock_image)
    assert isinstance(result, list), "Expected the result to be a list"
    assert all(isinstance(det, DetectionResult) for det in result), "All items should be DetectionResult instances"


def test_predict_with_image_path(yolo_detector: YoloV9ObjectDetector, mock_image_path: Path) -> None:
    """Test the predict method with a single image path."""
    result: list[DetectionResult] = yolo_detector.predict(str(mock_image_path))
    assert isinstance(result, list), "Expected the result to be a list"
    assert all(isinstance(det, DetectionResult) for det in result), "All items should be DetectionResult instances"


def test_predict_with_list_of_image_arrays(yolo_detector: YoloV9ObjectDetector, mock_image: np.ndarray) -> None:
    """Test the predict method with a list of image arrays."""
    # list of two identical image arrays
    images: list[np.ndarray] = [mock_image, mock_image]
    results: list[list[DetectionResult]] = yolo_detector.predict(images)
    assert isinstance(results, list), "Expected the result to be a list of lists"
    assert all(isinstance(res, list) for res in results), "Each result should be a list of DetectionResult"
    assert all(isinstance(det, DetectionResult) for res in results for det in res), (
        "All items should be DetectionResult instances"
    )


def test_predict_with_list_of_image_paths(yolo_detector: YoloV9ObjectDetector, mock_image_path: Path) -> None:
    """Test the predict method with a list of image paths."""
    # list of two identical image paths
    image_paths: list[str] = [str(mock_image_path), str(mock_image_path)]
    results: list[list[DetectionResult]] = yolo_detector.predict(image_paths)
    assert isinstance(results, list), "Expected the result to be a list of lists"
    assert all(isinstance(res, list) for res in results), "Each result should be a list of DetectionResult"
    assert all(isinstance(det, DetectionResult) for res in results for det in res), (
        "All items should be DetectionResult instances"
    )


def test_predict_with_invalid_image_path(yolo_detector: YoloV9ObjectDetector) -> None:
    """Test the predict method with an invalid image path."""
    invalid_path: str = "non_existent_image.jpg"
    with pytest.raises(ValueError, match=f"Failed to load image at path: {invalid_path}"):
        yolo_detector.predict(invalid_path)


@no_type_check
def test_predict_with_mixed_list(
    yolo_detector: YoloV9ObjectDetector, mock_image: np.ndarray, mock_image_path: Path
) -> None:
    """Test the predict method with a mixed list of image arrays and paths (should raise TypeError)."""
    mixed_input: list[object] = [mock_image, str(mock_image_path)]
    with pytest.raises(TypeError):
        yolo_detector.predict(mixed_input)


def test_predict_with_empty_list(yolo_detector: YoloV9ObjectDetector) -> None:
    """Test the predict method with an empty list."""
    results: list = yolo_detector.predict([])
    assert results == [], "Expected an empty list when input is empty"
