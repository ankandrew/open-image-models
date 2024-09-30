import numpy as np
import pytest

from open_image_models.detection.core.yolo_v9.preprocess import preprocess


@pytest.fixture(name="sample_image")
def _sample_image() -> np.ndarray:
    """Provides a sample image in BGR format for testing."""
    # Create a dummy image (BGR format) of size 640x480
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return img


@pytest.mark.parametrize(
    "img_size, expected_shape",
    [
        ((640, 640), (1, 3, 640, 640)),
        ((320, 320), (1, 3, 320, 320)),
        ((800, 800), (1, 3, 800, 800)),
    ],
)
def test_preprocess(sample_image: np.ndarray, img_size, expected_shape: tuple[int, int, int, int]) -> None:
    """Test preprocess function with different image sizes."""
    img, ratio, _ = preprocess(sample_image, img_size)
    assert img.shape == expected_shape, f"Expected shape {expected_shape}, but got {img.shape}"
    assert isinstance(ratio, tuple) and len(ratio) == 2, "Ratio must be a tuple of length 2."


def test_preprocess_output_range(sample_image: np.ndarray) -> None:
    """Ensure that the output image tensor is correctly normalized between 0 and 1."""
    img, _, _ = preprocess(sample_image, (640, 640))
    # The preprocessed image should be normalized to the range [0, 1]
    assert img.min() >= 0.0 and img.max() <= 1.0, "Preprocessed image values should be in the range [0, 1]"
