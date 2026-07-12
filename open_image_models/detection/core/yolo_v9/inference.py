import logging
import os
import pathlib
from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import onnxruntime as ort
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from open_image_models.detection.core.base import (
    DetectionResult,
    draw_detection_results,
    inspect_model_input_shape,
    resolve_image_inputs,
)
from open_image_models.detection.core.yolo_v9.postprocess import convert_to_detection_result
from open_image_models.detection.core.yolo_v9.preprocess import preprocess
from open_image_models.utils import measure_time, set_seed

# Setup logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _split_batch_predictions(predictions: np.ndarray, batch_size: int) -> list[np.ndarray]:
    if predictions.ndim == 3 and predictions.shape[0] == batch_size:
        return [predictions[index] for index in range(batch_size)]
    if predictions.ndim == 2 and predictions.shape[1] >= 7:
        return [predictions[predictions[:, 0] == index] for index in range(batch_size)]
    raise ValueError(f"Unexpected YOLOv9 output shape for batch size {batch_size}: {predictions.shape}")


class YoloV9Detector:
    """
    YoloV9-specific ONNX inference class for performing object detection using the Yolo v9 ONNX model.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        class_labels: list[str],
        *,
        conf_thresh: float | None = None,
        batch_size: int = 1,
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
    ) -> None:
        """
        Initializes the YOLOv9 detector with the specified model and inference device.

        Args:
            model_path: Path to the ONNX model file to use.
            class_labels: List of class labels corresponding to the class IDs.
            conf_thresh: Confidence threshold for filtering predictions. Defaults to 0.25.
            batch_size: Maximum inference batch size for dynamic-batch models.
            providers: Optional sequence of providers in order of decreasing precedence. If not specified, all available
            providers are used.
            sess_options: Advanced session options for ONNX Runtime.
        """
        self.conf_thresh = 0.25 if conf_thresh is None else conf_thresh
        self.class_labels = class_labels
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # Check if model path exists
        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at '{model_path}'")
        self.model_name = model_path.stem
        # Use all available providers if none are specified
        providers = providers or ort.get_available_providers()
        self.model = ort.InferenceSession(str(model_path), providers=providers, sess_options=sess_options)
        # Get input and output names from the model
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        input_shape = inspect_model_input_shape(self.model.get_inputs()[0].shape)
        if input_shape.image_size[0] != input_shape.image_size[1]:
            raise ValueError(f"Model only supports square images, but received shape: {input_shape.image_size}")
        self.img_size = input_shape.image_size
        self.batch_size = batch_size if input_shape.dynamic_batch else 1
        LOGGER.info("Using ONNX Runtime with %s provider(s)", providers)

    @overload
    def predict(self, images: np.ndarray) -> list[DetectionResult]: ...

    @overload
    def predict(self, images: list[np.ndarray]) -> list[list[DetectionResult]]: ...

    @overload
    def predict(self, images: str) -> list[DetectionResult]: ...

    @overload
    def predict(self, images: list[str]) -> list[list[DetectionResult]]: ...

    @overload
    def predict(self, images: os.PathLike[str]) -> list[DetectionResult]: ...

    @overload
    def predict(self, images: list[os.PathLike[str]]) -> list[list[DetectionResult]]: ...

    def predict(self, images: Any) -> list[DetectionResult] | list[list[DetectionResult]]:
        """
        Perform object detection on one or multiple images.

        Args:
            images: A single image as a numpy array, a single image path as a string, a list of images as numpy arrays,
                    or a list of image file paths.

        Returns:
            A list of DetectionResult for a single image input,
            or a list of lists of DetectionResult for multiple images.
        """
        loaded_images, is_single = resolve_image_inputs(images)
        results: list[list[DetectionResult]] = []
        for start in range(0, len(loaded_images), self.batch_size):
            results.extend(self._predict_batch(loaded_images[start : start + self.batch_size]))
        return results[0] if is_single else results

    def _predict(self, image: np.ndarray) -> list[DetectionResult]:
        return self._predict_batch([image])[0]

    def _predict_batch(self, images: list[np.ndarray]) -> list[list[DetectionResult]]:
        """
        Performs object detection on an image batch.

        Args:
            images: Input image frames in BGR format.

        Returns:
            Detection results for each image.
        """
        processed = [preprocess(image, self.img_size) for image in images]
        inputs = np.concatenate([item[0] for item in processed])
        try:
            predictions = self.model.run([self.output_name], {self.input_name: inputs})[0]
        # CoreML doesn't handle empty data scenario, so sometimes the end to end NMS might fail.
        # For more information see https://github.com/microsoft/onnxruntime/issues/20372
        # pylint: disable=broad-except
        except Exception as e:
            LOGGER.warning("An error occurred during model inference: %s", e)
            return [[] for _ in images]

        batch_predictions = _split_batch_predictions(np.asarray(predictions), len(images))
        return [
            convert_to_detection_result(
                predictions=image_predictions,
                class_labels=self.class_labels,
                ratio=ratio,
                padding=padding,
                score_threshold=self.conf_thresh,
            )
            for image_predictions, (_, ratio, padding) in zip(batch_predictions, processed, strict=True)
        ]

    def show_benchmark(self, num_runs: int = 1_000):
        """
        Display the model performance with a single random image.

        Args:
            num_runs: Number of times to run inference on the image for averaging.

        Displays:
            Model information and benchmark results in a formatted table.
        """
        # Set seed for generating same image
        set_seed(1337)
        # Generate a single random image
        image = np.random.randint(0, 256, (*self.img_size, 3), dtype=np.uint8)
        # Warm-up phase
        self._warm_up(image, num_runs=100)
        # Measure performance
        total_time_ms = self._benchmark_inference(image, num_runs)
        avg_time_ms = total_time_ms / num_runs
        fps = 1_000 / avg_time_ms if avg_time_ms > 0 else float("inf")
        # Display model information and benchmark results
        self._display_benchmark_results(avg_time_ms, fps, num_runs)

    def _warm_up(self, image: np.ndarray, num_runs: int = 100):
        """
        Warm-up phase to ensure any initial model setup is done before benchmarking.

        Args:
            image (np.ndarray): Sample image for model warm-up.
            num_runs (int, optional): Number of warm-up iterations. Default is 100.
        """
        LOGGER.info("Starting model warm-up with %d runs...", num_runs)
        for _ in range(num_runs):
            self.predict(image)
        LOGGER.info("Model warm-up completed.")

    def _benchmark_inference(self, image: np.ndarray, num_runs: int) -> float:
        """
        Measure the total inference time over a specified number of runs for benchmarking.

        Args:
            image (np.ndarray): Input image for inference benchmarking.
            num_runs (int): Number of benchmark iterations.

        Returns:
            float: Total inference time in milliseconds over all iterations.
        """
        LOGGER.info("Starting benchmark with %d runs...", num_runs)
        total_time_ms = 0.0
        for _ in range(num_runs):
            with measure_time() as timer:
                self.predict(image)
            total_time_ms += timer()
        LOGGER.info("Benchmark completed.")
        return total_time_ms

    def _display_benchmark_results(self, avg_time_ms: float, fps: float, num_runs: int):
        """
        Display benchmark results including average inference time and frames per second (FPS).

        This method presents the benchmarking results in a formatted table using the rich library,
        including the number of runs, average inference time, and the calculated FPS.

        Args:
            avg_time_ms (float): Average inference time in milliseconds.
            fps (float): Calculated frames per second based on average time.
            num_runs (int): Total number of benchmark runs.
        """
        console = Console()
        # Printing model details outside the table
        model_info = Panel(
            Text(f"Model: {self.model_name}\nProvider: {self.model.get_providers()}", style="bold green"),
            title="Model Information",
            border_style="bright_blue",
            expand=False,
        )
        console.print(model_info)
        # Creating the results table
        table = Table(title="YoloV9 Object Detector Performance", border_style="bright_blue")
        table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
        table.add_column("Value", justify="center", style="magenta")
        table.add_row("Number of Runs", str(num_runs))
        table.add_row("Average Time (ms)", f"{avg_time_ms:.2f}")
        table.add_row("Frames Per Second (FPS)", f"{fps:.2f}")
        # Display the table
        console.print(table)

    def display_predictions(self, image: np.ndarray) -> np.ndarray:
        """
        Run object detection on the input image and display the predictions on the image.

        Args:
            image: An input image as a numpy array.

        Returns:
            The image with bounding boxes and labels drawn on it.
        """
        detections: list[DetectionResult] = self.predict(image)
        return draw_detection_results(image, detections)
