"""File-only, cancellable video inference worker for the desktop demonstrator.

The worker owns capture/writer objects in its dedicated ``QThread``.  A final
video is never opened for writing: frames go to a sibling temporary file and
are published only after capture, inference and writer finalisation succeed.
This intentionally has no camera-device branch.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from fruit_ssod.detection import DetectionRecord, DetectorAdapter
from fruit_ssod.detection.adapter import validate_confidence_threshold, validate_nms_iou_threshold


SUPPORTED_VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".m4v"})


class VideoInferenceError(RuntimeError):
    """Actionable setup, decoding, inference, or publication failure."""


def _error(problem: str, cause: str, remediation: str) -> VideoInferenceError:
    return VideoInferenceError(
        f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."
    )


@dataclass(frozen=True)
class VideoInferenceSettings:
    """Immutable controls captured before the processing thread starts."""

    confidence: float = 0.25
    nms_iou: float = 0.50

    def __post_init__(self) -> None:
        confidence = validate_confidence_threshold(self.confidence)
        nms_iou = validate_nms_iou_threshold(self.nms_iou)
        if confidence is None or nms_iou is None:
            raise _error(
                "video inference controls are incomplete",
                "confidence or NMS IoU was omitted",
                "provide both confidence and NMS IoU values from 0 to 1",
            )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "nms_iou", nms_iou)


@dataclass(frozen=True)
class VideoSourceInfo:
    """Decoder metadata reported after the worker opens one local video file."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class VideoFrameResult:
    """One ordered display/result event; ``rgb_frame`` is a detached RGB copy."""

    frame_index: int
    rgb_frame: np.ndarray
    detections: tuple[DetectionRecord, ...]
    inference_fps: float
    source_fps: float


def resolve_video_path(path: str | Path) -> Path:
    """Resolve one local, supported video file without accepting devices/URLs."""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise _error(
            "selected video file was not found",
            f"{candidate} is not an existing regular file",
            "select a readable local video file",
        )
    if candidate.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise _error(
            "selected file is not a supported video",
            f"{candidate.name} has extension {candidate.suffix or 'none'}",
            "select MP4, MOV, AVI, MKV, or M4V video file",
        )
    return candidate.resolve()


def _annotate(frame_bgr: np.ndarray, detections: tuple[DetectionRecord, ...]) -> np.ndarray:
    """Draw canonical five-fruit records onto a private writer/display frame."""
    colors = {0: (45, 45, 220), 1: (25, 190, 240), 2: (35, 130, 245), 3: (100, 55, 210), 4: (35, 155, 120)}
    annotated = frame_bgr.copy()
    for item in detections:
        x1, y1, x2, y2 = (round(value) for value in item.xyxy)
        color = colors[item.class_id]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{item.class_name} {item.confidence:.2f}",
            (x1 + 2, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def _temporary_video_path(destination: Path) -> Path:
    """Return a same-volume temporary filename preserving a container suffix."""
    return destination.parent / f".{destination.stem}.processing-{uuid.uuid4().hex}{destination.suffix}"


def _publish_video(staging: Path, destination: Path) -> None:
    """Publish with an exclusive hard link, never overwriting an existing output.

    ``os.link`` is atomic and fails if the target exists, unlike replace-based
    publication.  It is available on supported local Windows NTFS volumes.  If
    a filesystem does not support it, fail safely and leave no final export.
    """
    if destination.exists():
        raise _error(
            "video export destination already exists",
            f"{destination} already exists",
            "choose a new output filename; existing videos are never overwritten",
        )
    try:
        os.link(staging, destination)
    except FileExistsError as error:
        raise _error(
            "video export destination was claimed by another export",
            f"{destination} appeared while processing was completing",
            "choose a different output filename and retry",
        ) from error
    except OSError as error:
        raise _error(
            "completed video could not be safely published",
            str(error),
            "choose a writable local NTFS output folder and a new filename",
        ) from error
    try:
        staging.unlink()
    except OSError as error:
        # The final is complete and valid; report the cleanup failure rather than
        # deleting a valid user output.  A later explicit cleanup can remove it.
        raise _error(
            "temporary processed video could not be removed",
            str(error),
            "keep the completed output and remove the hidden processing file manually",
        ) from error


def _fsync_staging_video(staging: Path) -> None:
    """Make a released staging video durable before its final-name publish."""
    # Windows rejects FlushFileBuffers for a read-only handle.  Open read/write
    # solely to make the completed writer data durable before the exclusive
    # final-name publication.
    with staging.open("r+b") as handle:
        os.fsync(handle.fileno())


class VideoInferenceWorker(QObject):
    """Decode/process one file in order and cooperatively pause or stop it."""

    opened = Signal(object)
    frame_completed = Signal(object, int, int)
    progress_changed = Signal(int, int)
    pause_changed = Signal(bool)
    stopped = Signal(int, int)
    completed = Signal(str, int, float)
    failed = Signal(str)
    finished = Signal(bool)

    def __init__(
        self,
        *,
        adapter: DetectorAdapter,
        video_path: str | Path,
        output_path: str | Path,
        settings: VideoInferenceSettings,
    ) -> None:
        super().__init__()
        if not isinstance(adapter, DetectorAdapter):
            raise _error(
                "no compatible detector is active",
                f"received {type(adapter).__name__}",
                "load a compatible five-fruit .pt model before processing video",
            )
        self._adapter = adapter
        self._video_path = resolve_video_path(video_path)
        self._output_path = Path(output_path).expanduser().resolve()
        if self._output_path.suffix.lower() != ".mp4":
            raise _error(
                "video export format is unsupported",
                f"{self._output_path.name} is not an .mp4 output",
                "save the processed video with an .mp4 filename",
            )
        if self._output_path.exists():
            raise _error(
                "video export destination already exists",
                f"{self._output_path} already exists",
                "choose a new output filename; existing videos are never overwritten",
            )
        self._settings = settings
        self._stop_requested = threading.Event()
        self._pause_condition = threading.Condition()
        self._paused = False
        # Publication is a one-way critical section.  A stop that wins before
        # this flag is set discards the staging file; a stop that arrives after
        # it is set cannot honestly report a cancellation because the final
        # filename may already have been atomically published.
        self._publication_lock = threading.Lock()
        self._publication_started = False

    def request_pause(self, paused: bool) -> None:
        """Set paused state from the GUI thread; wake a stopped/continued worker."""
        with self._pause_condition:
            if self._stop_requested.is_set():
                return
            self._paused = bool(paused)
            self._pause_condition.notify_all()
        self.pause_changed.emit(bool(paused))

    def request_stop(self) -> None:
        """Stop before final publication, or let an already-publishing run finish.

        The publication decision is protected by ``_publication_lock``.  Once
        it has been made, final output is no longer cancellable: reporting
        ``stopped`` at that point would be false if the atomic publish succeeds.
        """
        with self._publication_lock:
            if self._publication_started:
                return
            self._stop_requested.set()
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def _wait_until_resumed_or_stopped(self) -> bool:
        with self._pause_condition:
            while self._paused and not self._stop_requested.is_set():
                self._pause_condition.wait()
        return not self._stop_requested.is_set()

    def _begin_publication_if_not_stopped(self) -> bool:
        """Atomically choose cancellation or the irreversible final publish."""
        with self._publication_lock:
            if self._stop_requested.is_set():
                return False
            self._publication_started = True
            return True

    @Slot()
    def run(self) -> None:
        """Run file decoding/inference in frame order and publish only on success."""
        capture: cv2.VideoCapture | None = None
        writer: cv2.VideoWriter | None = None
        staging: Path | None = None
        processed = 0
        total = 0
        stopped = False
        started_at = perf_counter()
        try:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            if self._output_path.exists():
                raise _error(
                    "video export destination already exists",
                    f"{self._output_path} already exists",
                    "choose a new output filename; existing videos are never overwritten",
                )
            capture = cv2.VideoCapture(str(self._video_path))
            if not capture.isOpened():
                raise _error(
                    "video could not be opened",
                    f"OpenCV could not decode {self._video_path.name}",
                    "use a readable local MP4/MOV/AVI/MKV file with a supported codec",
                )
            total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not (fps > 0 and width > 0 and height > 0):
                raise _error(
                    "video metadata is invalid",
                    f"decoded fps={fps}, width={width}, height={height}",
                    "use a video with valid frame rate and dimensions",
                )
            self.opened.emit(VideoSourceInfo(self._video_path, total, fps, width, height))
            staging = _temporary_video_path(self._output_path)
            writer = cv2.VideoWriter(
                str(staging), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise _error(
                    "processed video writer could not be opened",
                    f"OpenCV could not create {staging.name}",
                    "choose a writable local folder and ensure MP4 codecs are available",
                )
            while True:
                if self._stop_requested.is_set() or not self._wait_until_resumed_or_stopped():
                    stopped = True
                    break
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if self._stop_requested.is_set():
                    stopped = True
                    break
                inference_started = perf_counter()
                detections = tuple(
                    self._adapter.predict(
                        frame_bgr,
                        confidence=self._settings.confidence,
                        nms_iou=self._settings.nms_iou,
                    )
                )
                elapsed = perf_counter() - inference_started
                annotated = _annotate(frame_bgr, detections)
                writer.write(annotated)
                processed += 1
                display_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB).copy()
                self.frame_completed.emit(
                    VideoFrameResult(
                        frame_index=processed,
                        rgb_frame=display_rgb,
                        detections=detections,
                        inference_fps=(1.0 / elapsed if elapsed > 0 else float("inf")),
                        source_fps=fps,
                    ),
                    processed,
                    total,
                )
                self.progress_changed.emit(processed, total)
            if self._stop_requested.is_set():
                stopped = True
            if writer is not None:
                writer.release()
                writer = None
            if capture is not None:
                capture.release()
                capture = None
            if stopped:
                self.stopped.emit(processed, total)
                return
            if processed == 0:
                raise _error(
                    "video contains no decodable frames",
                    f"{self._video_path.name} yielded zero frames",
                    "select a non-empty video encoded with a supported codec",
                )
            if total > 0 and processed != total:
                raise _error(
                    "video decoding ended before the advertised frame count",
                    f"processed {processed} of {total} frames",
                    "use an intact video file and retry; no partial export was published",
                )
            assert staging is not None
            # EOF is not yet a successful result.  Stop remains valid until
            # the one-way publication decision below, including after OpenCV
            # has released its capture/writer and while the staging file is
            # being synchronised.
            if self._stop_requested.is_set():
                stopped = True
                self.stopped.emit(processed, total)
                return
            _fsync_staging_video(staging)
            if self._stop_requested.is_set():
                stopped = True
                self.stopped.emit(processed, total)
                return
            if not self._begin_publication_if_not_stopped():
                stopped = True
                self.stopped.emit(processed, total)
                return
            _publish_video(staging, self._output_path)
            staging = None
            self.completed.emit(str(self._output_path), processed, perf_counter() - started_at)
        except VideoInferenceError as error:
            self.failed.emit(str(error))
        except Exception as error:
            self.failed.emit(
                str(
                    _error(
                        "video inference could not be completed",
                        str(error),
                        "verify the video, active model, available memory, and output folder before retrying",
                    )
                )
            )
        finally:
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()
            if staging is not None:
                try:
                    staging.unlink()
                except FileNotFoundError:
                    pass
            # ``finished`` is commonly connected to ``QThread.quit`` by GUI
            # callers.  That auto-connection can be queued to the GUI thread;
            # a caller that observes ``completed`` and immediately waits on
            # the thread would then block the queued quit delivery.  Request
            # termination directly from the worker's own thread first.  The
            # public signal remains for existing widgets and cleanup hooks.
            self.thread().quit()
            self.finished.emit(stopped)
