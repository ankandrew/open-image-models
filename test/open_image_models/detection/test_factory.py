from pathlib import Path

import pytest

from open_image_models.detection import factory

MODEL_NAME = "yolo-v9-t-256-license-plate-end2end"


def test_registered_model_rejects_metadata_overrides():
    with pytest.raises(ValueError, match="cannot override"):
        factory.create_detector(MODEL_NAME, backend="yolo_v9", class_labels=["plate"])


def test_local_model_requires_backend_and_labels(tmp_path):
    model_path = tmp_path / "custom.onnx"
    model_path.touch()

    with pytest.raises(ValueError, match="required for a local model"):
        factory.create_detector(model_path)


def test_local_model_must_exist():
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        factory.create_detector(Path("missing.onnx"), backend="rf_detr", class_labels=["vehicle"])
