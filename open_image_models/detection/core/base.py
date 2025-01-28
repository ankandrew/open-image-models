from dataclasses import dataclass
from typing import Any, Optional, Protocol

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

    @property
    def width(self) -> int:
        """Returns the width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Returns the height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Returns the area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """
        Returns the (x, y) coordinates of the center of the bounding box.
        """
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0

        return cx, cy

    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """
        Returns the intersection of this bounding box with another bounding box. If they do not intersect, returns None.
        """
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 > x1 and y2 > y1:
            return BoundingBox(x1, y1, x2, y2)

        return None

    def iou(self, other: "BoundingBox") -> float:
        """
        Computes the Intersection-over-Union (IoU) between this bounding box and another bounding box.
        """
        inter = self.intersection(other)

        if inter is None:
            return 0.0

        inter_area = inter.area
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def to_xywh(self) -> tuple[int, int, int, int]:
        """
        Converts bounding box to (x, y, width, height) format, where (x, y) is the top-left corner.
        """
        return self.x1, self.y1, self.width, self.height


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
