"""
Preprocess functions for YOLOv9.
"""

import cv2
import numpy as np


def preprocess(img: np.ndarray, img_size: tuple[int, int] | int):
    """
    Preprocess the input image for model inference.

    :param img: Input image in BGR format.
    :param img_size: Desired size to resize the image.
    :return: Preprocessed image tensor, resize ratio, and padding (dw, dh).
    """
    # Resize the input image to match training format
    im, ratio, (dw, dh) = letterbox(img, img_size)
    # HWC to CHW, BGR to RGB
    im = im.transpose((2, 0, 1))[::-1]
    # 0 - 255 to 0.0 - 1.0
    im = im / 255.0
    # Model precision is FP32
    im = im.astype(np.float32)
    # Add batch dimension
    im = np.expand_dims(im, 0)
    return im, ratio, (dw, dh)


def letterbox(
    im: np.ndarray,
    new_shape: tuple[int, int] | int = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Simplified letterbox function with fixed behavior for YOLOv9 preprocessing.

    Resizes and pads the input image to the desired size while maintaining aspect ratio.
    """
    shape = im.shape[:2]  # current shape [height, width]

    # Convert integer new_shape to a tuple (new_shape, new_shape)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calculate the scaling ratio and resize the image
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Calculate new unpadded dimensions and padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2  # divide padding into 2 sides
    dh = (new_shape[0] - new_unpad[1]) / 2

    # Resize the image to the new unpadded dimensions
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add padding to maintain the new shape with the specified color
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, (r, r), (dw, dh)
