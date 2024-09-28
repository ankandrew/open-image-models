from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class DetectionResult:
    label: str
    confidence: float
    bounding_box: BoundingBox


class ObjectDetector(Protocol):
    def predict(self, image: np.ndarray) -> list[DetectionResult]:
        """
        Run object detection on the input image.

        Args:
            image: An input image as a numpy array.

        Returns:
            A list of DetectionResult containing detected objects information.
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
