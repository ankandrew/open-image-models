import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Optional, Protocol, cast

import cv2
import numpy as np

ClassLabels = Sequence[str] | Mapping[int, str]
"""Class labels as a contiguous sequence or explicit class-ID mapping."""


def normalize_class_labels(class_labels: ClassLabels) -> dict[int, str]:
    """
    Normalize class labels into an explicit class-ID mapping.

    Args:
        class_labels: Contiguous labels or a mapping from model class IDs to labels.

    Returns:
        A class-ID-to-label dictionary.

    Raises:
        ValueError: If no labels are supplied or a class ID or label is invalid.
    """
    if isinstance(class_labels, str):
        raise ValueError("class_labels must contain at least one label")

    labels = dict(class_labels.items()) if isinstance(class_labels, Mapping) else dict(enumerate(class_labels))
    if not labels:
        raise ValueError("class_labels must contain at least one label")
    if any(isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0 for class_id in labels):
        raise ValueError("class label IDs must be non-negative integers")
    if any(not isinstance(label, str) or not label for label in labels.values()):
        raise ValueError("class labels must be non-empty strings")
    return labels


@dataclass(frozen=True)
class BoundingBox:  # pylint: disable=too-many-public-methods
    """
    Represents an axis-aligned 2D bounding box defined by two corner points.
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
        """
        Returns:
            The horizontal distance from `x1` to `x2`.
        """
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """
        Returns:
            The vertical distance from `y1` to `y2`.
        """
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """
        Returns:
            The bounding box area, or zero if the box is empty.
        """
        return max(0, self.width) * max(0, self.height)

    @property
    def aspect_ratio(self) -> float:
        """
        Returns:
            The width-to-height ratio, or zero if the box is empty.
        """
        return self.width / self.height if not self.is_empty else 0.0

    @property
    def is_empty(self) -> bool:
        """
        Returns:
            `True` if either dimension is zero or negative, otherwise `False`.
        """
        return self.width <= 0 or self.height <= 0

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """
        Returns:
            The coordinates as `(x1, y1, x2, y2)`.
        """
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        """
        Returns:
            The center coordinates as `(center_x, center_y)`.
        """
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0

        return cx, cy

    @classmethod
    def from_xywh(cls, x: int, y: int, width: int, height: int) -> "BoundingBox":
        """
        Creates a bounding box from top-left coordinates, width, and height.

        Args:
            x: X-coordinate of the left edge.
            y: Y-coordinate of the top edge.
            width: Width of the bounding box.
            height: Height of the bounding box.

        Returns:
            A bounding box in `(x1, y1, x2, y2)` format.
        """
        return cls(x, y, x + width, y + height)

    @classmethod
    def from_cxcywh(cls, center_x: float, center_y: float, width: float, height: float) -> "BoundingBox":
        """
        Creates a bounding box from center coordinates, width, and height.

        Coordinates are rounded outward so the integer box contains the full floating-point region.

        Args:
            center_x: X-coordinate of the center.
            center_y: Y-coordinate of the center.
            width: Width of the bounding box.
            height: Height of the bounding box.

        Returns:
            A bounding box with integer coordinates.
        """
        half_width = width / 2
        half_height = height / 2
        return cls(
            floor(center_x - half_width),
            floor(center_y - half_height),
            ceil(center_x + half_width),
            ceil(center_y + half_height),
        )

    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """
        Computes the intersection with another bounding box.

        Args:
            other: The bounding box to intersect with this box.

        Returns:
            The positive-area intersection, or `None` if the boxes do not overlap.
        """
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 > x1 and y2 > y1:
            return BoundingBox(x1, y1, x2, y2)

        return None

    def intersects(self, other: "BoundingBox") -> bool:
        """
        Checks whether this bounding box intersects another.

        Args:
            other: The bounding box to test.

        Returns:
            `True` if the boxes have a positive-area intersection, otherwise `False`.
        """
        return self.intersection(other) is not None

    def contains_point(self, x: float, y: float) -> bool:
        """
        Checks whether a point lies within this bounding box.

        Args:
            x: X-coordinate of the point.
            y: Y-coordinate of the point.

        Returns:
            `True` if the point is inside the box or on its boundary, otherwise `False`.
        """
        return not self.is_empty and self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def contains(self, other: "BoundingBox") -> bool:
        """
        Checks whether this bounding box fully contains another.

        Args:
            other: The bounding box to test.

        Returns:
            `True` if both boxes are non-empty and this box contains all of `other`, otherwise `False`.
        """
        return (
            not self.is_empty
            and not other.is_empty
            and self.x1 <= other.x1
            and self.y1 <= other.y1
            and other.x2 <= self.x2
            and other.y2 <= self.y2
        )

    def iou(self, other: "BoundingBox") -> float:
        """
        Computes the Intersection-over-Union (IoU) with another bounding box.

        Args:
            other: The bounding box to compare with this box.

        Returns:
            The IoU in the range `[0.0, 1.0]`, or zero if the union has no positive area.
        """
        inter = self.intersection(other)

        if inter is None:
            return 0.0

        inter_area = inter.area
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def intersection_over_area(self, other: "BoundingBox") -> float:
        """
        Computes the intersection divided by this bounding box's area.

        Args:
            other: The bounding box to compare with this box.

        Returns:
            The fraction of this box covered by `other`, or zero if this box is empty or the boxes do not intersect.
        """
        inter = self.intersection(other)
        return inter.area / self.area if inter is not None and self.area > 0 else 0.0

    def enclosing(self, other: "BoundingBox") -> "BoundingBox":
        """
        Computes the smallest bounding box containing both boxes.

        Args:
            other: The bounding box to enclose with this box.

        Returns:
            A bounding box spanning this box and `other`.
        """
        return BoundingBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )

    def to_xywh(self) -> tuple[int, int, int, int]:
        """
        Converts the bounding box to top-left, width, and height format.

        Returns:
            The bounding box as `(x, y, width, height)`.
        """
        return self.x1, self.y1, self.width, self.height

    def to_cxcywh(self) -> tuple[float, float, float, float]:
        """
        Converts the bounding box to center, width, and height format.

        Returns:
            The bounding box as `(center_x, center_y, width, height)`.
        """
        return *self.center, float(self.width), float(self.height)

    def as_slices(self) -> tuple[slice, slice]:
        """
        Converts the bounding box to NumPy-compatible image slices.

        NumPy excludes the stop coordinate, so pixels at `x2` and `y2` are not included.

        Returns:
            A `(rows, columns)` tuple equivalent to `(slice(y1, y2), slice(x1, x2))`.
        """
        return slice(self.y1, self.y2), slice(self.x1, self.x2)

    def __iter__(self) -> Iterator[int]:
        """
        Iterates over the bounding box coordinates.

        Returns:
            An iterator over `(x1, y1, x2, y2)`.
        """
        return iter(self.xyxy)

    def clamp(self, max_width: int, max_height: int) -> "BoundingBox":
        """
        Clamps the bounding box coordinates to frame boundaries.

        Args:
            max_width: Maximum x-coordinate, normally the frame width.
            max_height: Maximum y-coordinate, normally the frame height.

        Returns:
            A bounding box whose coordinates lie within `[0, max_width]` and `[0, max_height]`.
        """
        return BoundingBox(
            x1=max(0, min(self.x1, max_width)),
            y1=max(0, min(self.y1, max_height)),
            x2=max(0, min(self.x2, max_width)),
            y2=max(0, min(self.y2, max_height)),
        )

    def is_inside(self, frame_width: int, frame_height: int) -> bool:
        """
        Checks whether all coordinates lie within frame boundaries.

        This method only checks coordinate bounds; it does not require the box to have positive area. Use `is_valid`
        when both conditions are needed.

        Args:
            frame_width: Width of the frame.
            frame_height: Height of the frame.

        Returns:
            `True` if all coordinates lie within the frame, otherwise `False`.
        """
        return self.x1 >= 0 and self.y1 >= 0 and self.x2 <= frame_width and self.y2 <= frame_height

    def is_valid(self, frame_width: int, frame_height: int) -> bool:
        """
        Checks whether the bounding box is non-empty and inside a frame.

        Args:
            frame_width: Width of the frame.
            frame_height: Height of the frame.

        Returns:
            `True` if the coordinates are ordered, have positive area, and lie inside the frame boundaries,
            otherwise `False`.
        """
        return not self.is_empty and self.is_inside(frame_width, frame_height)


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

    @classmethod
    def from_detection_data(
        cls,
        bbox_data: tuple[int, int, int, int],
        confidence: float,
        label: str,
    ) -> "DetectionResult":
        """
        Creates a `DetectionResult` instance from bounding box data, confidence, and a class label.

        Args:
            bbox_data: Bounding box coordinates as `(x1, y1, x2, y2)`.
            confidence: Detection confidence score.
            label: Detected class label.

        Returns:
            A detection result containing the supplied data.
        """
        bounding_box = BoundingBox(*bbox_data)
        return cls(label, confidence, bounding_box)


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

    def display_predictions(self, image: np.ndarray) -> np.ndarray:
        """
        Run object detection on the input image and display the predictions on the image.

        Args:
            image: An input image as a numpy array.

        Returns:
            The image with bounding boxes and labels drawn on it.
        """


@dataclass(frozen=True)
class ModelInputShape:
    """Relevant dimensions read from a detector's NCHW input shape."""

    image_size: tuple[int, int]
    dynamic_batch: bool


def inspect_model_input_shape(input_shape: Sequence[Any]) -> ModelInputShape:
    """
    Validates a detector input shape and determines its batching capability.

    Args:
        input_shape: ONNX model input shape in NCHW format.

    Returns:
        Static image dimensions and whether the batch dimension is dynamic.

    Raises:
        ValueError: If the shape is not NCHW, does not use three channels, has
            dynamic spatial dimensions, or has a fixed batch size other than one.
    """
    if len(input_shape) != 4:
        raise ValueError(f"Expected model input shape in NCHW format, got {input_shape}")

    batch_size, channels, height, width = input_shape
    if channels != 3:
        raise ValueError(f"Expected model input with 3 channels, got {channels}")
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError(f"Expected static input height and width, got {input_shape}")
    if isinstance(batch_size, int) and batch_size != 1:
        raise ValueError(f"Expected a dynamic batch or fixed batch size 1, got {batch_size}")

    return ModelInputShape(image_size=(height, width), dynamic_batch=not isinstance(batch_size, int))


def resolve_image_inputs(images: Any) -> tuple[list[np.ndarray], bool]:
    """
    Normalize supported detector inputs into loaded image arrays.

    Args:
        images: A single image array/path or a list of image arrays/paths.

    Returns:
        A tuple of loaded BGR images and whether the original input was a single image.
    """
    if isinstance(images, np.ndarray):
        return [images], True
    if isinstance(images, str | os.PathLike):
        return [_load_image(images)], True
    if isinstance(images, list):
        if all(isinstance(img, np.ndarray) for img in images):
            return cast(list[np.ndarray], images), False
        if all(isinstance(img, str | os.PathLike) for img in images):
            return [_load_image(img) for img in images], False
        raise TypeError("List must contain either all numpy arrays or all image file paths.")
    raise TypeError("Input must be a numpy array, a list of numpy arrays, or a list of image file paths.")


def draw_detection_results(image: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
    """
    Draw detection results on an image.

    Args:
        image: Input image to mutate.
        detections: Detection results to draw.

    Returns:
        The input image with bounding boxes and labels drawn on it.
    """
    for detection in detections:
        bbox = detection.bounding_box
        label = f"{detection.label}: {detection.confidence:.2f}"
        cv2.rectangle(image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            image,
            (bbox.x1, bbox.y1 - text_height - baseline),
            (bbox.x1 + text_width, bbox.y1),
            (0, 255, 0),
            thickness=cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (bbox.x1, bbox.y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    return image


def _load_image(image_path: str | os.PathLike[str]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image at path: {image_path}")
    return image
