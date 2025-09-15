"""
Open Image Models package.
"""

import warnings

__all__: list[str] = []

try:
    from open_image_models.detection.pipeline.license_plate import LicensePlateDetector  # noqa: F401
except ImportError:
    warnings.warn(
        "[open_image_models] LicensePlateDetector won't be available to use."
        " Make sure you to install with `pip install open-image-models[onnx]`.",
        stacklevel=2,
    )
else:
    __all__.append("LicensePlateDetector")
