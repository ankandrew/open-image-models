import numpy as np

from open_image_models.detection.core.base import BoundingBox, DetectionResult


def convert_to_detection_result(
    boxes: np.ndarray,
    logits: np.ndarray,
    class_labels: list[str],
    image_size: tuple[int, int],
    score_threshold: float = 0.5,
    num_select: int = 300,
) -> list[DetectionResult]:
    """
    Convert exported RF-DETR outputs into DetectionResult objects.

    Args:
        boxes:
            Normalized cxcywh boxes with shape (Q, 4) or (1, Q, 4).
        logits:
            Raw class logits with shape (Q, C), (Q, C + 1), (1, Q, C), or (1, Q, C + 1).
        class_labels:
            Class labels corresponding to user-visible class IDs.
        image_size:
            Original image size as (height, width).
        score_threshold:
            Minimum sigmoid confidence to include.
        num_select:
            Maximum number of query/class pairs to consider.

    Returns:
        Detection results ordered by descending confidence.
    """
    boxes, logits = _prepare_outputs(boxes, logits, class_labels, image_size, score_threshold)

    if num_select <= 0:
        return []

    query_indexes, class_ids, selected_scores = _select_top_scores(logits, num_select)

    # Discard low-confidence results before box conversion and object creation
    keep = selected_scores > score_threshold
    if not np.any(keep):
        return []

    query_indexes = query_indexes[keep]
    class_ids = class_ids[keep]
    selected_scores = selected_scores[keep]

    xyxy_boxes = _scale_boxes(boxes[query_indexes], image_size)

    return _create_detection_results(
        boxes=xyxy_boxes,
        class_ids=class_ids,
        scores=selected_scores,
        class_labels=class_labels,
        image_size=image_size,
    )


def _prepare_outputs(
    boxes: np.ndarray,
    logits: np.ndarray,
    class_labels: list[str],
    image_size: tuple[int, int],
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = _remove_batch_dimension(np.asarray(boxes), "boxes")
    logits = _remove_batch_dimension(np.asarray(logits), "logits")

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape (Q, 4), got {boxes.shape}.")
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (Q, C), got {logits.shape}.")
    if boxes.shape[0] != logits.shape[0]:
        raise ValueError(f"Boxes contain {boxes.shape[0]} queries, but logits contain {logits.shape[0]} queries.")
    if any(dimension <= 0 for dimension in image_size):
        raise ValueError(f"Image dimensions must be positive, got {image_size}.")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError(f"score_threshold must be between 0 and 1, got {score_threshold}.")

    if logits.shape[1] == len(class_labels) + 1:
        logits = logits[:, :-1]
    elif logits.shape[1] != len(class_labels):
        raise ValueError(f"Model returned {logits.shape[1]} class slots, but {len(class_labels)} labels were supplied.")

    return boxes, logits


def _remove_batch_dimension(values: np.ndarray, output_name: str) -> np.ndarray:
    if values.ndim != 3:
        return values
    if values.shape[0] != 1:
        raise ValueError(f"Only single-image output is supported, but {output_name} have batch size {values.shape[0]}.")
    return values[0]


def _select_top_scores(logits: np.ndarray, num_select: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_logits = logits.reshape(-1)

    if flat_logits.size == 0 or num_select <= 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    selected_count = min(num_select, flat_logits.size)

    # Select the largest values without constructing -flat_logits
    partition_index = flat_logits.size - selected_count
    selected_indexes = np.argpartition(flat_logits, partition_index)[partition_index:]
    # Sort the selected subset into descending order
    selected_indexes = selected_indexes[np.argsort(flat_logits[selected_indexes])[::-1]]

    num_classes = logits.shape[1]
    query_indexes = selected_indexes // num_classes
    class_ids = selected_indexes % num_classes

    selected_logits = flat_logits[selected_indexes].astype(np.float32, copy=False)
    selected_logits = np.clip(selected_logits, -88.0, 88.0)
    selected_scores = np.reciprocal(np.float32(1.0) + np.exp(-selected_logits))

    return query_indexes, class_ids, selected_scores


def _scale_boxes(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    image_height, image_width = image_size

    boxes = boxes.astype(np.float32, copy=False)
    xyxy_boxes = _box_cxcywh_to_xyxy(boxes)

    scale = np.array([image_width, image_height, image_width, image_height], dtype=xyxy_boxes.dtype)

    xyxy_boxes *= scale
    np.clip(xyxy_boxes, 0, scale, out=xyxy_boxes)

    return xyxy_boxes


def _create_detection_results(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    class_labels: list[str],
    image_size: tuple[int, int],
) -> list[DetectionResult]:
    image_height, image_width = image_size
    results: list[DetectionResult] = []

    for bbox, class_id, score in zip(boxes, class_ids, scores, strict=True):
        bounding_box = BoundingBox(x1=int(bbox[0]), y1=int(bbox[1]), x2=int(bbox[2]), y2=int(bbox[3]))

        if not bounding_box.is_valid(image_width, image_height):
            continue

        results.append(
            DetectionResult(label=class_labels[int(class_id)], confidence=float(score), bounding_box=bounding_box)
        )

    return results


def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    center_x = boxes[:, 0:1]
    center_y = boxes[:, 1:2]
    width = np.maximum(boxes[:, 2:3], 0.0)
    height = np.maximum(boxes[:, 3:4], 0.0)

    half_width = width / 2
    half_height = height / 2

    return np.concatenate(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        axis=1,
    )
