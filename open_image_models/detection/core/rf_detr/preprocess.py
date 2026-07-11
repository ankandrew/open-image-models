import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
    """
    Prepare a BGR image for an exported RF-DETR ONNX model.

    Args:
        image: Input image in BGR format.
        img_size: Target model input size as (height, width).

    Returns:
        A float32 NCHW batch with ImageNet normalization.
    """
    # Prepare input image
    img_height, img_width = img_size
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    image_array = image_resized.astype(np.float32) / 255.0
    # Normalize
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    # Convert to NCHW format
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0).astype(np.float32)
