import numpy as np

from open_image_models.detection.core.base import BoundingBox, DetectionResult


def convert_to_detection_result(
    predictions: np.ndarray,
    class_labels: list[str],
    ratio: tuple[float, float],
    padding: tuple[float, float],
    score_threshold: float = 0.5,
) -> list[DetectionResult]:
    """
    Convert raw model output into a list of DetectionResult objects, with adjustments
    for padding and scaling to the original image size.

    Args:
        predictions: The concatenated array of the form
                     [X, selected_boxes, selected_categories, selected_scores].
        class_labels: List of class labels corresponding to the class IDs.
        ratio: Scaling ratio used during preprocessing.
        padding: Tuple of padding values (dw, dh) added during preprocessing.
        score_threshold: Minimum confidence score to include a detection result.

    Returns:
        A list of DetectionResult objects.
    """
    results = []
    # Extract bounding box coordinates
    bboxes = predictions[:, 1:5]
    # Convert to int for indexing class labels
    class_ids = predictions[:, 5].astype(int)
    # Confidence scores
    scores = predictions[:, 6]

    for bbox, class_id, score in zip(bboxes, class_ids, scores, strict=False):
        # Only include results that meet the score threshold
        if score < score_threshold:
            continue

        # Adjust bounding box from scaled image back to original image size
        bbox[0] = (bbox[0] - padding[0]) / ratio[0]
        bbox[1] = (bbox[1] - padding[1]) / ratio[1]
        bbox[2] = (bbox[2] - padding[0]) / ratio[0]
        bbox[3] = (bbox[3] - padding[1]) / ratio[1]

        # Map class_id to label if available
        label = class_labels[class_id] if class_id < len(class_labels) else str(class_id)

        # Create BoundingBox and DetectionResult instances
        bounding_box = BoundingBox(x1=int(bbox[0]), y1=int(bbox[1]), x2=int(bbox[2]), y2=int(bbox[3]))
        detection_result = DetectionResult(
            label=label,
            confidence=float(score),
            bounding_box=bounding_box,
        )
        results.append(detection_result)

    return results
