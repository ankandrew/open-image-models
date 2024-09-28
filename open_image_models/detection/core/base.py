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
            A list of dictionaries containing detection results,
            where each dictionary includes 'label', 'confidence',
            and 'bounding_box' (tuple of x_min, y_min, x_max, y_max).
        """
