"""
License Plate object detection pipeline.
"""

import logging
from collections.abc import Sequence

import onnxruntime as ort

from open_image_models.detection.core.hub import PlateDetectorModel, download_model
from open_image_models.detection.core.yolo_v9.inference import YoloV9ObjectDetector

# Setup logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class LicensePlateDetector(YoloV9ObjectDetector):
    """
    Specialized detector for license plates using YoloV9 model.
    Inherits from YoloV9ObjectDetector and sets up license plate specific configuration.
    """

    def __init__(
        self,
        detection_model: PlateDetectorModel,
        conf_thresh: float = 0.25,
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions = None,
    ) -> None:
        """
        Initializes the LicensePlateDetector with the specified detection model and inference device.

        Args:
            detection_model: Detection model to use, see `PlateDetectorModel`.
            conf_thresh: Confidence threshold for filtering predictions.
            providers: Optional sequence of providers in order of decreasing precedence. If not specified, all available
            providers are used.
            sess_options: Advanced session options for ONNX Runtime.
        """
        # Download model if needed
        detector_model_path = download_model(detection_model)
        super().__init__(
            model_path=detector_model_path,
            conf_thresh=conf_thresh,
            class_labels=["License Plate"],
            providers=providers,
            sess_options=sess_options,
        )
        LOGGER.info("Initialized LicensePlateDetector with model %s", detector_model_path)
