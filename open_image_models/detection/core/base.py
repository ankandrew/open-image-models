from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """
    Represents a bounding box with top-left and bottom-right coordinates.
    """

    x1: int
    """X-coordinate of the top-left corner"""
    y1: int
    """Y-coordinate of the top-left corner"""
    x2: int
    """X-coordinate of the bottom-right corner"""
    y2: int
    """Y-coordinate of the bottom-right corner"""


@dataclass(frozen=True)
class DetectionResult:
    """
    Represents the result of an object detection.
    """

    label: str
    """Detected object label"""
    confidence: float
    """Confidence score of the detection"""
    bounding_box: BoundingBox
    """Bounding box of the detected object"""


class ObjectDetector(Protocol):
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

    def show_benchmark(self, num_runs: int = 10) -> None:
        """
        Display the benchmark results of the model with a single random image.

        Args:
            num_runs: Number of times to run inference on the image for averaging.

        Displays:
            Model information and benchmark results in a formatted table.
        """

    def display_predictions(self, image: np.ndarray) -> np.ndarray:
        """
        Run object detection on the input image and display the predictions on the image.

        Args:
            image: An input image as a numpy array.

        Returns:
            The image with bounding boxes and labels drawn on it.
        """
