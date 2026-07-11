import numpy as np

from open_image_models.detection.core.base import BoundingBox
from open_image_models.detection.core.rf_detr.postprocess import convert_to_detection_result
from open_image_models.detection.core.rf_detr.preprocess import preprocess


def test_rf_detr_preprocess_returns_normalized_nchw_batch():
    image = np.full((8, 10, 3), 255, dtype=np.uint8)

    result = preprocess(image, img_size=(4, 6))

    assert result.shape == (1, 3, 4, 6)
    assert result.dtype == np.float32


def test_rf_detr_postprocess_converts_normalized_boxes_to_source_image_size():
    boxes = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    logits = np.array(
        [
            [-10.0, 6.0],
            [-10.0, -10.0],
        ],
        dtype=np.float32,
    )

    results = convert_to_detection_result(
        boxes=boxes,
        logits=logits,
        class_labels=["background", "vehicle"],
        image_size=(100, 200),
        score_threshold=0.9,
    )

    assert len(results) == 1
    assert results[0].label == "vehicle"
    assert results[0].bounding_box == BoundingBox(50, 25, 150, 75)
