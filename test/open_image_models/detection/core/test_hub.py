"""
Tests for ONNX hub module.
"""

from http import HTTPStatus
from typing import get_args

import pytest
import requests

from open_image_models.detection.core.hub import DETECTION_MODELS, DetectionModelName


def test_registered_models_match_model_name_literal():
    assert set(DETECTION_MODELS) == set(get_args(DetectionModelName))


@pytest.mark.parametrize("model_name", DETECTION_MODELS)
def test_model_and_config_urls(model_name):
    """
    Test that the registered model URL is valid.
    """
    model_url = DETECTION_MODELS[model_name].url
    response = requests.head(model_url, timeout=5, allow_redirects=True)
    assert response.status_code == HTTPStatus.OK, f"URL {model_url} is not accessible, got {response.status_code}"
