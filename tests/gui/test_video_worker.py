"""Regression coverage for the file-only video inference worker."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread

from fruit_ssod.detection import DetectorAdapter
from fruit_ssod.gui.workers import video_worker as video_worker_module
from fruit_ssod.gui.workers.video_worker import (
    VideoInferenceSettings,
    VideoInferenceWorker,
    resolve_video_path,
)
from fruit_ssod.gui.widgets.video_view import VideoInferencePage


FIXTURE_VIDEO = Path(__file__).parents[1] / "fixtures" / "video" / "tiny_video.mp4"


def _make_video(path: Path, *, frames: int = 4) -> Path:
    """Create a deterministic MP4 fixture if a source checkout lacks binary assets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 24))
    assert writer.isOpened()
    for value in range(frames):
        frame = np.full((24, 32, 3), value * 45, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def tiny_video() -> Path:
    # The repository fixture is used when present. The fallback keeps the
    # source test portable for source distributions that omit binary assets.
    return FIXTURE_VIDEO if FIXTURE_VIDEO.is_file() else _make_video(FIXTURE_VIDEO)


class RecordingVideoAdapter(DetectorAdapter):
    def __init__(self) -> None:
        self.means: list[int] = []
        self.thread_ids: list[int] = []

    def predict(self, image: object, *, confidence: float | None = None, nms_iou: float | None = None) -> tuple[object, ...]:
        assert isinstance(image, np.ndarray)
        assert confidence == 0.35 and nms_iou == 0.65
        self.means.append(int(image.mean()))
        self.thread_ids.append(threading.get_ident())
        return ()


class BlockingVideoAdapter(RecordingVideoAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def predict(self, image: object, *, confidence: float | None = None, nms_iou: float | None = None) -> tuple[object, ...]:
        self.started.set()
        assert self.release.wait(3)
        return super().predict(image, confidence=confidence, nms_iou=nms_iou)


class PauseAfterFirstAdapter(RecordingVideoAdapter):
    """Keep the page's pause request observable before frame two begins."""

    def __init__(self) -> None:
        super().__init__()
        self.first_complete = threading.Event()
        self.release_first = threading.Event()

    def predict(self, image: object, *, confidence: float | None = None, nms_iou: float | None = None) -> tuple[object, ...]:
        if not self.first_complete.is_set():
            self.first_complete.set()
            assert self.release_first.wait(3)
        return super().predict(image, confidence=confidence, nms_iou=nms_iou)


def _start(worker: VideoInferenceWorker) -> QThread:
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.start()
    return thread


def test_worker_preserves_frame_order_reports_progress_and_publishes_complete_video(
    tmp_path: Path, tiny_video: Path, qtbot: object
) -> None:
    adapter = RecordingVideoAdapter()
    destination = tmp_path / "complete.mp4"
    worker = VideoInferenceWorker(
        adapter=adapter,
        video_path=tiny_video,
        output_path=destination,
        settings=VideoInferenceSettings(confidence=0.35, nms_iou=0.65),
    )
    frame_indices: list[int] = []
    progress: list[tuple[int, int]] = []
    completed: list[tuple[str, int, float]] = []
    failures: list[str] = []
    worker.frame_completed.connect(lambda result, *_args: frame_indices.append(result.frame_index))
    worker.progress_changed.connect(lambda current, total: progress.append((current, total)))
    worker.completed.connect(lambda path, count, elapsed: completed.append((path, count, elapsed)))
    worker.failed.connect(failures.append)
    thread = _start(worker)
    qtbot.waitUntil(lambda: bool(completed) or bool(failures), timeout=5_000)  # type: ignore[attr-defined]
    assert thread.wait(5_000)

    assert failures == []

    assert frame_indices == [1, 2, 3, 4]
    assert [item[0] for item in progress] == [1, 2, 3, 4]
    assert all(total == 4 for _current, total in progress)
    assert adapter.means == sorted(adapter.means)
    assert all(thread_id != threading.get_ident() for thread_id in adapter.thread_ids)
    assert completed[0][0] == str(destination.resolve())
    assert completed[0][1] == 4
    assert destination.is_file() and destination.stat().st_size > 0
    capture = cv2.VideoCapture(str(destination))
    assert capture.isOpened()
    assert sum(1 for _ in iter(lambda: capture.read()[0], False)) == 4
    capture.release()
    assert not list(tmp_path.glob(".complete.processing-*.mp4"))


def test_stop_after_current_frame_discards_temporary_and_never_publishes_final(
    tmp_path: Path, tiny_video: Path, qtbot: object
) -> None:
    adapter = BlockingVideoAdapter()
    destination = tmp_path / "cancelled.mp4"
    worker = VideoInferenceWorker(
        adapter=adapter,
        video_path=tiny_video,
        output_path=destination,
        settings=VideoInferenceSettings(confidence=0.35, nms_iou=0.65),
    )
    stopped: list[tuple[int, int]] = []
    frames: list[int] = []
    worker.stopped.connect(lambda current, total: stopped.append((current, total)))
    worker.frame_completed.connect(lambda result, *_args: frames.append(result.frame_index))
    thread = _start(worker)
    qtbot.waitUntil(adapter.started.is_set, timeout=3_000)  # type: ignore[attr-defined]
    worker.request_stop()
    adapter.release.set()
    qtbot.waitUntil(lambda: bool(stopped), timeout=5_000)  # type: ignore[attr-defined]
    assert thread.wait(5_000)

    # A detector call may finish after stop is requested, but it is not written
    # or surfaced as a completed frame and no partial final path exists.
    assert frames == [1]
    assert stopped == [(1, 4)]
    assert not destination.exists()
    assert not list(tmp_path.glob(".cancelled.processing-*.mp4"))


def test_stop_after_eof_before_publish_discards_staging_and_never_emits_completed(
    tmp_path: Path, tiny_video: Path, qtbot: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EOF/release is still cancellable until the final-name decision is made."""
    destination = tmp_path / "eof_cancelled.mp4"
    worker = VideoInferenceWorker(
        adapter=RecordingVideoAdapter(),
        video_path=tiny_video,
        output_path=destination,
        settings=VideoInferenceSettings(confidence=0.35, nms_iou=0.65),
    )
    real_fsync = video_worker_module._fsync_staging_video

    def stop_at_finalisation(staging: Path) -> None:
        # ``run`` has already released capture and writer before it invokes
        # this hook.  The second stop check must win before publication.
        worker.request_stop()
        real_fsync(staging)

    monkeypatch.setattr(video_worker_module, "_fsync_staging_video", stop_at_finalisation)
    stopped: list[tuple[int, int]] = []
    completed: list[tuple[str, int, float]] = []
    failures: list[str] = []
    worker.stopped.connect(lambda current, total: stopped.append((current, total)))
    worker.completed.connect(lambda path, frames, elapsed: completed.append((path, frames, elapsed)))
    worker.failed.connect(failures.append)
    thread = _start(worker)
    qtbot.waitUntil(lambda: bool(stopped) or bool(completed) or bool(failures), timeout=5_000)  # type: ignore[attr-defined]
    assert thread.wait(5_000)

    assert stopped == [(4, 4)]
    assert completed == []
    assert failures == []
    assert not destination.exists()
    assert not list(tmp_path.glob(".eof_cancelled.processing-*.mp4"))


def test_stop_after_publish_does_not_misreport_a_completed_video_as_stopped(
    tmp_path: Path, tiny_video: Path, qtbot: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once publication starts, cancellation no longer changes the final state."""
    destination = tmp_path / "published_then_stop.mp4"
    worker = VideoInferenceWorker(
        adapter=RecordingVideoAdapter(),
        video_path=tiny_video,
        output_path=destination,
        settings=VideoInferenceSettings(confidence=0.35, nms_iou=0.65),
    )
    real_publish = video_worker_module._publish_video

    def publish_then_stop(staging: Path, final_path: Path) -> None:
        real_publish(staging, final_path)
        # This models a Stop click racing just after the atomic final-name
        # publication.  It must not convert success into a false "stopped".
        worker.request_stop()

    monkeypatch.setattr(video_worker_module, "_publish_video", publish_then_stop)
    stopped: list[tuple[int, int]] = []
    completed: list[tuple[str, int, float]] = []
    failures: list[str] = []
    worker.stopped.connect(lambda current, total: stopped.append((current, total)))
    worker.completed.connect(lambda path, frames, elapsed: completed.append((path, frames, elapsed)))
    worker.failed.connect(failures.append)
    thread = _start(worker)
    qtbot.waitUntil(lambda: bool(completed) or bool(failures), timeout=5_000)  # type: ignore[attr-defined]
    assert thread.wait(5_000)

    assert stopped == []
    assert failures == []
    assert completed and completed[0][0] == str(destination.resolve())
    assert destination.is_file() and destination.stat().st_size > 0
    assert not list(tmp_path.glob(".published_then_stop.processing-*.mp4"))


def test_existing_final_output_is_preserved_and_staging_is_cleaned(
    tmp_path: Path, tiny_video: Path
) -> None:
    destination = tmp_path / "existing.mp4"
    destination.write_bytes(b"do not overwrite")
    with pytest.raises(Exception, match="already exists"):
        VideoInferenceWorker(
            adapter=RecordingVideoAdapter(),
            video_path=tiny_video,
            output_path=destination,
            settings=VideoInferenceSettings(),
        )
    assert destination.read_bytes() == b"do not overwrite"
    assert not list(tmp_path.glob(".existing.processing-*.mp4"))


def test_video_path_rejects_nonfile_and_does_not_accept_camera_like_input(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="not found"):
        resolve_video_path(tmp_path / "0")


def test_video_page_pauses_resumes_and_saves_only_after_complete_run(
    tmp_path: Path, tiny_video: Path, qtbot: object
) -> None:
    adapter = PauseAfterFirstAdapter()
    page = VideoInferencePage(model_provider=lambda: adapter, start_allowed=lambda: True)
    qtbot.addWidget(page)  # type: ignore[attr-defined]
    page.show()
    destination = tmp_path / "page_complete.mp4"
    assert page.set_video(tiny_video)
    assert page.set_output_path(destination)
    page._confidence.setValue(0.35)
    page._nms_iou.setValue(0.65)
    assert page.start_processing()
    qtbot.waitUntil(adapter.first_complete.is_set, timeout=3_000)  # type: ignore[attr-defined]
    page.toggle_pause()
    adapter.release_first.set()
    qtbot.waitUntil(lambda: page._pause_button.text() == "Resume", timeout=3_000)  # type: ignore[attr-defined]
    # The first call may have completed but the paused worker must not consume
    # further frames until Resume is explicitly selected.
    qtbot.waitUntil(lambda: len(adapter.means) == 1, timeout=3_000)  # type: ignore[attr-defined]
    assert not destination.exists()
    page.toggle_pause()
    qtbot.waitUntil(lambda: not page.is_running, timeout=5_000)  # type: ignore[attr-defined]
    assert page._finished_successfully is True
    assert destination.is_file() and destination.stat().st_size > 0


def test_video_page_shutdown_timeout_recovers_only_after_its_active_worker_exits(
    tmp_path: Path, tiny_video: Path, qtbot: object
) -> None:
    adapter = BlockingVideoAdapter()
    page = VideoInferencePage(model_provider=lambda: adapter, start_allowed=lambda: True)
    qtbot.addWidget(page)  # type: ignore[attr-defined]
    page.show()
    first_destination = tmp_path / "timed_out_close.mp4"
    second_destination = tmp_path / "recovered_run.mp4"
    assert page.set_video(tiny_video)
    assert page.set_output_path(first_destination)
    page._confidence.setValue(0.35)
    page._nms_iou.setValue(0.65)
    assert page.start_processing()
    qtbot.waitUntil(adapter.started.is_set, timeout=3_000)  # type: ignore[attr-defined]

    assert page.shutdown(wait_ms=1) is False
    assert page.is_running is True
    assert page._shutdown_requested is True
    assert page.set_video(tiny_video) is False
    assert page.start_processing() is False

    adapter.release.set()
    qtbot.waitUntil(lambda: not page.is_running, timeout=3_000)  # type: ignore[attr-defined]
    qtbot.waitUntil(lambda: not page._bridges, timeout=3_000)  # type: ignore[attr-defined]
    assert page._thread is None
    assert page._worker is None
    assert page._bridge is None
    assert page._shutdown_requested is False
    assert page._open_button.isEnabled() is True
    assert page._play_button.isEnabled() is True
    assert not first_destination.exists()

    assert page.set_video(tiny_video)
    assert page.set_output_path(second_destination)
    assert page.start_processing()
    qtbot.waitUntil(lambda: not page.is_running, timeout=5_000)  # type: ignore[attr-defined]
    assert page._finished_successfully is True
    assert second_destination.is_file() and second_destination.stat().st_size > 0


def test_video_page_successful_shutdown_remains_closing_after_worker_exit(
    tmp_path: Path, tiny_video: Path, qtbot: object
) -> None:
    adapter = BlockingVideoAdapter()
    page = VideoInferencePage(model_provider=lambda: adapter, start_allowed=lambda: True)
    qtbot.addWidget(page)  # type: ignore[attr-defined]
    page.show()
    destination = tmp_path / "successful_close.mp4"
    assert page.set_video(tiny_video)
    assert page.set_output_path(destination)
    page._confidence.setValue(0.35)
    page._nms_iou.setValue(0.65)
    assert page.start_processing()
    qtbot.waitUntil(adapter.started.is_set, timeout=3_000)  # type: ignore[attr-defined]

    release = threading.Timer(0.05, adapter.release.set)
    release.start()
    try:
        assert page.shutdown(wait_ms=3_000) is True
    finally:
        release.cancel()
        adapter.release.set()
    qtbot.waitUntil(lambda: not page._bridges, timeout=3_000)  # type: ignore[attr-defined]
    assert page._shutdown_requested is True
    assert page._shutdown_timed_out is False
    assert page.set_video(tiny_video) is False
    assert page.start_processing() is False
    assert not destination.exists()
