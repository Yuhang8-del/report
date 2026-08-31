"""Real-time camera capture and Ultralytics inference for the desktop GUI."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot


class CameraInferenceError(RuntimeError):
    """Actionable camera/model setup error suitable for the GUI."""


def _error(problem: str, cause: str, remediation: str) -> CameraInferenceError:
    return CameraInferenceError(
        f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."
    )


FIVE_CLASS_NAMES = (
    "Apple",
    "Banana",
    "Orange",
    "Strawberry",
    "Pineapple",
)
ELEVEN_CLASS_NAMES = FIVE_CLASS_NAMES + (
    "Avocado",
    "Blueberry",
    "Cherry",
    "Kiwi",
    "Mango",
    "Rockmelon",
)


@dataclass(frozen=True)
class CameraModelProfile:
    """One selectable real-time detector with an explicit expected taxonomy."""

    label: str
    weights_path: Path
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        path = Path(self.weights_path).expanduser()
        if not self.label.strip():
            raise _error("empty model name", "the model profile has no display name", "provide a readable model name")
        if path.suffix.lower() != ".pt":
            raise _error("unsupported model format", f"{path.name} is not a .pt file", "select Ultralytics .pt weights")
        if not self.class_names or any(not name.strip() for name in self.class_names):
            raise _error("invalid model taxonomy", "class names are empty or missing", "provide the complete class list")
        object.__setattr__(self, "weights_path", path.resolve())


@dataclass(frozen=True)
class CameraInferenceSettings:
    """Validated real-time controls captured when a camera run starts."""

    confidence: float = 0.25
    nms_iou: float = 0.50
    width: int = 1280
    height: int = 720
    image_size: int = 640
    target_fps: int = 30
    device: str | int = 0

    def __post_init__(self) -> None:
        for name, value in (("confidence", self.confidence), ("NMS IoU", self.nms_iou)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise _error(f"invalid {name}", "the value is not finite", "enter a value between 0 and 1")
            if not 0.0 <= float(value) <= 1.0:
                raise _error(f"{name} is out of range", "the value is outside 0 to 1", "enter a value between 0 and 1")
        for name, value in (
            ("frame width", self.width),
            ("frame height", self.height),
            ("model input size", self.image_size),
            ("target frame rate", self.target_fps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise _error(f"invalid {name}", f"received {value!r}", "use a positive integer")


@dataclass(frozen=True)
class CameraDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraSourceInfo:
    device_index: int
    width: int
    height: int
    source_fps: float
    model_label: str
    class_count: int


@dataclass(frozen=True)
class CameraFrameResult:
    frame_index: int
    rgb_frame: np.ndarray
    detections: tuple[CameraDetection, ...]
    inference_ms: float
    inference_fps: float
    display_fps: float

    @property
    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in self.detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return counts


CaptureFactory = Callable[[int, int], Any]
ModelFactory = Callable[[Path], Any]


def _capture_backends() -> tuple[int, ...]:
    if os.name == "nt":
        return (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
    return (cv2.CAP_ANY,)


def _new_capture(index: int, backend: int) -> Any:
    return cv2.VideoCapture(index, backend)


def _open_capture(
    index: int,
    *,
    width: int,
    height: int,
    capture_factory: CaptureFactory = _new_capture,
) -> Any:
    """Open one device with Windows-friendly backends and configure resolution."""
    attempted: list[int] = []
    for backend in _capture_backends():
        attempted.append(backend)
        capture = capture_factory(index, backend)
        if capture is not None and capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        if capture is not None:
            capture.release()
    raise _error(
        f"camera {index} could not be opened",
        f"OpenCV backends {attempted} all failed",
        "check the camera connection and Windows privacy permissions, close other camera apps, then refresh",
    )


def probe_camera_devices(
    *,
    max_devices: int = 6,
    capture_factory: CaptureFactory = _new_capture,
) -> tuple[int, ...]:
    """Return readable camera indices; every temporary handle is released."""
    if isinstance(max_devices, bool) or not isinstance(max_devices, int) or max_devices <= 0:
        raise _error("invalid scan range", f"received {max_devices!r}", "use a positive integer")
    found: list[int] = []
    for index in range(max_devices):
        capture = None
        try:
            capture = _open_capture(
                index,
                width=640,
                height=480,
                capture_factory=capture_factory,
            )
            ok, frame = capture.read()
            if ok and isinstance(frame, np.ndarray) and frame.size > 0:
                found.append(index)
        except CameraInferenceError:
            pass
        finally:
            if capture is not None:
                capture.release()
    return tuple(found)


def _names_mapping(names: object) -> dict[int, str]:
    if isinstance(names, Mapping):
        mapping = {int(key): str(value) for key, value in names.items()}
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        mapping = {index: str(value) for index, value in enumerate(names)}
    else:
        raise _error("model taxonomy is missing", "the weights do not provide names", "use the packaged project weights")
    return mapping


def _to_numpy(value: object) -> np.ndarray:
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    return np.asarray(value)


def normalize_camera_result(raw_result: object, profile: CameraModelProfile) -> tuple[CameraDetection, ...]:
    """Convert one Ultralytics result without the five-class-only GUI contract."""
    names = _names_mapping(getattr(raw_result, "names", None))
    expected = dict(enumerate(profile.class_names))
    if names != expected:
        raise _error(
            "model taxonomy does not match the selected profile",
            f"expected {expected!r}, received {names!r}",
            "select the correct packaged 5-class or 11-class weights",
        )
    boxes = getattr(raw_result, "boxes", None)
    if boxes is None:
        return ()
    xyxy = _to_numpy(getattr(boxes, "xyxy", ()))
    confidences = _to_numpy(getattr(boxes, "conf", ()))
    classes = _to_numpy(getattr(boxes, "cls", ()))
    if not (len(xyxy) == len(confidences) == len(classes)):
        raise _error("incomplete model output", "box, confidence and class counts differ", "check model compatibility")
    detections: list[CameraDetection] = []
    for box, confidence, class_id_value in zip(xyxy, confidences, classes):
        class_id = int(class_id_value)
        coordinates = tuple(float(value) for value in box.tolist())
        if len(coordinates) != 4 or class_id not in names:
            raise _error("invalid detection box", "the class or coordinate format is invalid", "use a compatible Ultralytics detector")
        detections.append(
            CameraDetection(
                class_id=class_id,
                class_name=names[class_id],
                confidence=float(confidence),
                xyxy=coordinates,
            )
        )
    return tuple(detections)


class CameraInferenceWorker(QObject):
    """Own camera and model resources inside one background QThread."""

    opened = Signal(object)
    frame_ready = Signal(object)
    stopped = Signal(int)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        *,
        profile: CameraModelProfile,
        device_index: int,
        settings: CameraInferenceSettings,
        capture_factory: CaptureFactory = _new_capture,
        model_factory: ModelFactory | None = None,
    ) -> None:
        super().__init__()
        if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
            raise _error("invalid camera index", f"received {device_index!r}", "select a non-negative camera index")
        self._profile = profile
        self._device_index = device_index
        self._settings = settings
        self._capture_factory = capture_factory
        self._model_factory = model_factory or self._default_model_factory
        self._stop_requested = threading.Event()

    @staticmethod
    def _default_model_factory(weights: Path) -> Any:
        from ultralytics import YOLO

        return YOLO(str(weights))

    def request_stop(self) -> None:
        self._stop_requested.set()

    @Slot()
    def run(self) -> None:
        capture = None
        frame_index = 0
        previous_emitted_at: float | None = None
        failed_reads = 0
        try:
            if not self._profile.weights_path.is_file():
                raise _error(
                    "camera model was not found",
                    str(self._profile.weights_path),
                    "restore the packaged weights in the models folder",
                )
            model = self._model_factory(self._profile.weights_path)
            names = _names_mapping(getattr(model, "names", None))
            if names != dict(enumerate(self._profile.class_names)):
                raise _error("model taxonomy mismatch", f"received {names!r}", "select the correct packaged model")
            capture = _open_capture(
                self._device_index,
                width=self._settings.width,
                height=self._settings.height,
                capture_factory=self._capture_factory,
            )
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or self._settings.width
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._settings.height
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            self.opened.emit(
                CameraSourceInfo(
                    device_index=self._device_index,
                    width=actual_width,
                    height=actual_height,
                    source_fps=source_fps if source_fps > 0 else 0.0,
                    model_label=self._profile.label,
                    class_count=len(self._profile.class_names),
                )
            )
            while not self._stop_requested.is_set():
                frame_started_at = perf_counter()
                ok, frame_bgr = capture.read()
                if not ok or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
                    failed_reads += 1
                    if failed_reads >= 8:
                        raise _error(
                            "camera reads failed repeatedly",
                            "the device returned no valid frames",
                            "reconnect the camera, lower the resolution, or close other apps using it",
                        )
                    self._stop_requested.wait(0.03)
                    continue
                failed_reads = 0
                inference_started = perf_counter()
                raw_results = model.predict(
                    frame_bgr,
                    conf=self._settings.confidence,
                    iou=self._settings.nms_iou,
                    imgsz=self._settings.image_size,
                    device=self._settings.device,
                    verbose=False,
                )
                if not raw_results:
                    raise _error("model returned no result", "the inference result is empty", "check the model and camera frame")
                inference_ms = (perf_counter() - inference_started) * 1000.0
                detections = normalize_camera_result(raw_results[0], self._profile)
                frame_index += 1
                now = perf_counter()
                display_fps = (
                    1.0 / (now - previous_emitted_at)
                    if previous_emitted_at is not None and now > previous_emitted_at
                    else 0.0
                )
                previous_emitted_at = now
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
                self.frame_ready.emit(
                    CameraFrameResult(
                        frame_index=frame_index,
                        rgb_frame=rgb_frame,
                        detections=detections,
                        inference_ms=inference_ms,
                        inference_fps=(1000.0 / inference_ms if inference_ms > 0 else 0.0),
                        display_fps=display_fps,
                    )
                )
                frame_elapsed = perf_counter() - frame_started_at
                remaining = 1.0 / self._settings.target_fps - frame_elapsed
                if remaining > 0:
                    self._stop_requested.wait(remaining)
            self.stopped.emit(frame_index)
        except CameraInferenceError as error:
            self.failed.emit(str(error))
        except Exception as error:
            self.failed.emit(
                str(
                    _error(
                        "live inference terminated unexpectedly",
                        str(error),
                        "check the camera, model file, CUDA environment and GPU memory, then retry",
                    )
                )
            )
        finally:
            if capture is not None:
                capture.release()
            self.thread().quit()
            self.finished.emit()
