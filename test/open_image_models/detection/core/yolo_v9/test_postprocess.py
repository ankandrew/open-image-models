import numpy as np
import pytest

from open_image_models.detection.core.base import BoundingBox
from open_image_models.detection.core.yolo_v9.postprocess import convert_to_detection_result


@pytest.fixture(name="sample_predictions")
def _sample_predictions() -> np.ndarray:
    """
    Provides a sample prediction array for testing.

    Predictions format: [batch_id, x1, y1, x2, y2, class_id, score]
    """
    return np.array(
        [
            [0, 50, 60, 200, 220, 0, 0.9],  # High confidence detection for class 0
            [0, 150, 160, 300, 320, 1, 0.7],  # High confidence detection for class 1
            [0, 250, 260, 400, 420, 2, 0.4],  # Low confidence detection for class 2 (should be filtered out)
        ]
    )


@pytest.fixture(name="class_labels")
def _class_labels() -> list[str]:
    """Provides a list of class labels."""
    return ["person", "car", "dog"]


@pytest.fixture(name="scaling_ratio")
def _scaling_ratio() -> tuple[float, float]:
    """Provides a sample scaling ratio used during preprocessing."""
    return 0.5, 0.5  # Scaling applied by a factor of 0.5 on both axes


@pytest.fixture(name="padding_values")
def _padding_values() -> tuple[int, int]:
    """Provides a sample padding tuple used during preprocessing."""
    return 10, 10  # Padding added on both axes


@pytest.mark.parametrize(
    "score_threshold, expected_labels",
    [
        # Threshold at 0.5, only two detections should pass
        (0.5, ["person", "car"]),
        # Threshold at 0.8, only one detection should pass
        (0.8, ["person"]),
        # Threshold at 0.3, all detections should pass
        (0.3, ["person", "car", "dog"]),
    ],
)
def test_convert_to_detection_result(
    sample_predictions: np.ndarray,
    class_labels: list[str],
    scaling_ratio: tuple[float, float],
    padding_values: tuple[int, int],
    score_threshold: float,
    expected_labels: list[str],
) -> None:
    """Test the conversion of predictions to DetectionResult with different score thresholds."""
    results = convert_to_detection_result(
        predictions=sample_predictions,
        class_labels=class_labels,
        ratio=scaling_ratio,
        padding=padding_values,
        score_threshold=score_threshold,
    )
    # Check that the number of results matches the expected number of labels
    assert len(results) == len(expected_labels), f"Expected {len(expected_labels)} results, but got {len(results)}"
    # Check that labels match the expected labels
    for result, expected_label in zip(results, expected_labels, strict=False):
        assert result.label == expected_label, f"Expected label {expected_label}, but got {result.label}"
    # Check the confidence scores
    for result in results:
        assert result.confidence >= score_threshold, f"Expected score >= {score_threshold}, but got {result.confidence}"


def test_bounding_box_adjustment(
    sample_predictions: np.ndarray,
    class_labels: list[str],
    scaling_ratio: tuple[float, float],
    padding_values: tuple[int, int],
) -> None:
    """Test if bounding boxes are correctly adjusted based on ratio and padding."""
    results = convert_to_detection_result(
        predictions=sample_predictions,
        class_labels=class_labels,
        ratio=scaling_ratio,
        padding=padding_values,
        score_threshold=0.5,
    )
    # Expected adjusted bounding boxes (unpadding and scaling to original size)
    expected_bboxes = [
        BoundingBox(x1=80, y1=100, x2=380, y2=420),
        BoundingBox(x1=280, y1=300, x2=580, y2=620),
    ]
    for result, expected_bbox in zip(results, expected_bboxes, strict=False):
        bbox = result.bounding_box
        assert (
            bbox.x1 == expected_bbox.x1 and bbox.y1 == expected_bbox.y1
        ), f"Expected bbox x1={expected_bbox.x1}, y1={expected_bbox.y1}, but got x1={bbox.x1}, y1={bbox.y1}"
        assert (
            bbox.x2 == expected_bbox.x2 and bbox.y2 == expected_bbox.y2
        ), f"Expected bbox x2={expected_bbox.x2}, y2={expected_bbox.y2}, but got x2={bbox.x2}, y2={bbox.y2}"


@pytest.mark.parametrize(
    "class_labels, expected_label",
    [
        # Known class label
        (["person", "car", "dog"], "dog"),
        # Out of bounds class label should return class_id as string
        (["person", "car"], "2"),
    ],
)
def test_class_label_mapping(
    class_labels: list[str],
    scaling_ratio: tuple[float, float],
    padding_values: tuple[int, int],
    expected_label: str,
) -> None:
    """Test if the correct label is mapped or class_id returned when label is out of bounds."""
    predictions = np.array(
        [
            [0, 50, 60, 200, 220, 2, 0.9],  # High confidence detection for class_id 2
        ]
    )
    results = convert_to_detection_result(
        predictions=predictions,
        class_labels=class_labels,
        ratio=scaling_ratio,
        padding=padding_values,
        score_threshold=0.5,
    )
    assert results[0].label == expected_label, f"Expected label {expected_label}, but got {results[0].label}"


def test_no_results_below_threshold(
    sample_predictions: np.ndarray,
    class_labels: list[str],
    scaling_ratio: tuple[float, float],
    padding_values: tuple[int, int],
) -> None:
    """Test that no DetectionResult is returned if all scores are below the threshold."""
    results = convert_to_detection_result(
        predictions=sample_predictions,
        class_labels=class_labels,
        ratio=scaling_ratio,
        padding=padding_values,
        score_threshold=1.0,  # Threshold is higher than all scores
    )
    # No results should be returned
    assert isinstance(results, list), "Expected a list!"
    assert len(results) == 0, "Expected no results, but got some results"
