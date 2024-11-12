"""
Tests for ONNX hub module.
"""

from http import HTTPStatus

import pytest
import requests

from open_image_models.detection.core.hub import AVAILABLE_ONNX_MODELS


@pytest.mark.parametrize("model_name", AVAILABLE_ONNX_MODELS.keys())
def test_model_and_config_urls(model_name):
    """
    Test to check if the model URLs of AVAILABLE_ONNX_MODELS are valid.
    """
    model_url = AVAILABLE_ONNX_MODELS[model_name]
    response = requests.get(model_url, timeout=5, allow_redirects=True)
    assert response.status_code == HTTPStatus.OK, f"URL {model_url} is not accessible, got {response.status_code}"
