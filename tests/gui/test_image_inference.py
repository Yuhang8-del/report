"""pytest-qt coverage for file-only image inference and result export."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PIL import Image

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.detection import DetectionRecord, DetectorAdapter
from fruit_ssod.gui import result_exporter
from fruit_ssod.gui.result_exporter import ResultExportError, export_inference_results
from fruit_ssod.gui.widgets.image_view import ImageInferencePage
from fruit_ssod.gui.workers.image_worker import ImageInferenceResult, ImageInferenceSettings, ImageInferenceWorker
from PySide6.QtCore import QThread


def _image(path: Path, color: tuple[int, int, int] = (240, 220, 180)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color).save(path)
    return path


class RecordingAdapter(DetectorAdapter):
    """Pure-Python adapter proving that inference executes in the worker thread."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, float | None, float | None, int]] = []

    def predict(
        self, image: object, *, confidence: float | None = None, nms_iou: float | None = None
    ) -> tuple[DetectionRecord, ...]:
        assert isinstance(image, Path)
        self.calls.append((image, confidence, nms_iou, threading.get_ident()))
        return (
            DetectionRecord(
                class_id=0,
                class_name="Apple",
                confidence=0.91,
                xyxy=(5.0, 6.0, 50.0, 45.0),
                is_unknown=False,
                source_model="fixture.pt",
                registry=DEFAULT_CLASS_REGISTRY,
            ),
        )


class SecondRoundBlockingAdapter(RecordingAdapter):
    """Block only the replacement run, keeping its controls observable in test."""

    def __init__(self) -> None:
        super().__init__()
        self.second_round_started = threading.Event()
        self.release_second_round = threading.Event()

    def predict(
        self, image: object, *, confidence: float | None = None, nms_iou: float | None = None
    ) -> tuple[DetectionRecord, ...]:
        if confidence == 0.70:
            self.second_round_started.set()
            assert self.release_second_round.wait(3)
        return super().predict(image, confidence=confidence, nms_iou=nms_iou)


class ShutdownQueuedSuccessAdapter(RecordingAdapter):
    """Keep one predict call in flight until shutdown has invalidated its run."""

    def __init__(self) -> None:
        super().__init__()
        self.predict_started = threading.Event()
        self.release_predict = threading.Event()

    def predict(
        self, image: object, *, confidence: float | None = None, nms_iou: float | None = None
    ) -> tuple[DetectionRecord, ...]:
        self.predict_started.set()
        assert self.release_predict.wait(3)
        return super().predict(image, confidence=confidence, nms_iou=nms_iou)


def _result(image: Path) -> ImageInferenceResult:
    return ImageInferenceResult(
        image_path=image.resolve(), detections=(), latency_ms=1.0, confidence=0.25, nms_iou=0.50
    )


def test_worker_runs_prediction_off_the_gui_thread_and_passes_controls(tmp_path: Path, qtbot: object) -> None:
    """The model call happens in QThread and carries both visible controls."""
    image = _image(tmp_path / "apple.png")
    adapter = RecordingAdapter()
    worker = ImageInferenceWorker(
        adapter=adapter,
        image_paths=[image],
        settings=ImageInferenceSettings(confidence=0.35, nms_iou=0.65),
    )
    results: list[object] = []
    worker.image_completed.connect(lambda result, *_args: results.append(result))
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)

    with qtbot.waitSignal(worker.finished, timeout=3_000):  # type: ignore[attr-defined]
        thread.start()
    assert thread.wait(3_000)

    assert len(results) == 1
    result = results[0]
    assert result.image_path == image.resolve()  # type: ignore[union-attr]
    assert result.class_counts == {"Apple": 1}  # type: ignore[union-attr]
    assert adapter.calls == [(image.resolve(), 0.35, 0.65, adapter.calls[0][3])]
    assert adapter.calls[0][3] != threading.get_ident()


def test_batch_page_previews_previous_next_and_exports_artifacts(tmp_path: Path, qtbot: object) -> None:
    """A batch keeps stable preview order and emits annotated PNG, CSV, and JSON."""
    first = _image(tmp_path / "a" / "apple.png")
    second = _image(tmp_path / "b" / "apple.png", (200, 180, 100))
    adapter = RecordingAdapter()
    page = ImageInferencePage(model_provider=lambda: adapter, mode="batch", start_allowed=lambda: True)
    qtbot.addWidget(page)  # type: ignore[attr-defined]
    page.show()

    assert page.set_images([first, second]) is True
    assert page.selected_paths == (first.resolve(), second.resolve())
    page._navigate(1)
    assert page.findChild(type(page._image_view), "imagePreview").image_path == second.resolve()  # type: ignore[union-attr]
    assert page.start_inference() is True
    qtbot.waitUntil(lambda: not page.is_running, timeout=3_000)  # type: ignore[attr-defined]
    assert len(page.results) == 2
    manifest = page.export_results(tmp_path / "export")
    assert manifest is not None
    assert len(manifest.annotated_images) == 2
    assert all(path.is_file() for path in manifest.annotated_images)
    assert manifest.manifest_path.is_file()
    with manifest.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    payload = json.loads(manifest.json_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "fruit_ssod_image_inference_export"
    assert [item["confidence_threshold"] for item in payload["results"]] == [0.25, 0.25]


def test_new_image_run_clears_prior_results_and_only_exports_its_finished_round(tmp_path: Path, qtbot: object) -> None:
    """A blocked replacement run cannot expose or export the preceding run's data."""
    first = _image(tmp_path / "first.png")
    second = _image(tmp_path / "second.png", (180, 210, 120))
    adapter = SecondRoundBlockingAdapter()
    page = ImageInferencePage(model_provider=lambda: adapter, mode="single", start_allowed=lambda: True)
    qtbot.addWidget(page)  # type: ignore[attr-defined]
    page.show()

    assert page.set_images([first]) is True
    assert page.start_inference() is True
    qtbot.waitUntil(lambda: not page.is_running, timeout=3_000)  # type: ignore[attr-defined]
    qtbot.waitUntil(page._export_button.isEnabled, timeout=3_000)  # type: ignore[attr-defined]
    assert [result.image_path for result in page.results] == [first.resolve()]

    assert page.set_images([second]) is True
    page._confidence.setValue(0.70)
    assert page.start_inference() is True
    qtbot.waitUntil(adapter.second_round_started.is_set, timeout=3_000)  # type: ignore[attr-defined]
    assert page._export_button.isEnabled() is False
    assert page.results == ()
    assert page.export_results(tmp_path / "must_not_exist") is None
    assert not (tmp_path / "must_not_exist").exists()

    adapter.release_second_round.set()
    qtbot.waitUntil(lambda: not page.is_running, timeout=3_000)  # type: ignore[attr-defined]
    qtbot.waitUntil(page._export_button.isEnabled, timeout=3_000)  # type: ignore[attr-defined]
    manifest = page.export_results(tmp_path / "second_round_export")
    assert manifest is not None
    payload = json.loads(manifest.json_path.read_text(encoding="utf-8"))
    assert [Path(item["image_path"]) for item in payload["results"]] == [second.resolve()]
    assert [item["confidence_threshold"] for item in payload["results"]] == [0.70]


def test_shutdown_discards_success_queued_while_gui_waits(tmp_path: Path, qtbot: object) -> None:
    """A success emitted during shutdown must not repopulate or become exportable."""
    image = _image(tmp_path / "apple.png")
    adapter = ShutdownQueuedSuccessAdapter()
    page = ImageInferencePage(model_provider=lambda: adapter, mode="single", start_allowed=lambda: True)
    qtbot.addWidget(page)  # type: ignore[attr-defined]
    page.show()
    assert page.set_images([image])
    assert page.start_inference()
    qtbot.waitUntil(adapter.predict_started.is_set, timeout=3_000)  # type: ignore[attr-defined]
    release = threading.Timer(0.05, adapter.release_predict.set)
    release.start()
    try:
        assert page.shutdown(wait_ms=3_000) is True
    finally:
        release.cancel()
        adapter.release_predict.set()
    # Pump events queued by the worker while shutdown was blocked in QThread.wait.
    qtbot.waitUntil(lambda: not page._bridges, timeout=3_000)  # type: ignore[attr-defined]
    # A successful close stays closed; only a refused, timed-out close may
    # recover into a reusable page once its worker has exited.
    assert page._shutdown_requested is True
    assert page.results == ()
    assert page._round_finished_successfully is False
    assert page._export_button.isEnabled() is False
    assert page.export_results(tmp_path / "after_shutdown") is None
    assert not (tmp_path / "after_shutdown").exists()


def test_shutdown_timeout_late_thread_exit_recovers_page_for_a_clean_new_run(tmp_path: Path, qtbot: object) -> None:
    """A refused close releases late worker ownership without reviving stale output."""
    first = _image(tmp_path / "first.png")
    second = _image(tmp_path / "second.png", (180, 210, 120))
    adapter = ShutdownQueuedSuccessAdapter()
    page = ImageInferencePage(model_provider=lambda: adapter, mode="single", start_allowed=lambda: True)
    qtbot.addWidget(page)  # type: ignore[attr-defined]
    page.show()
    busy_states: list[bool] = []
    page.busy_changed.connect(busy_states.append)

    assert page.set_images([first])
    assert page.start_inference()
    qtbot.waitUntil(adapter.predict_started.is_set, timeout=3_000)  # type: ignore[attr-defined]
    # The worker is intentionally still blocked, so close is refused.  Its run
    # id is invalidated at this point, while the bridge remains the active
    # owner responsible for late cleanup.
    assert page.shutdown(wait_ms=1) is False
    assert page.is_running is True
    assert page._shutdown_requested is True

    adapter.release_predict.set()
    qtbot.waitUntil(lambda: not page.is_running, timeout=3_000)  # type: ignore[attr-defined]
    qtbot.waitUntil(lambda: not page._bridges, timeout=3_000)  # type: ignore[attr-defined]
    assert page._thread is None
    assert page._worker is None
    assert page._signal_bridge is None
    assert page._shutdown_requested is False
    assert page.results == ()
    assert page._round_finished_successfully is False
    assert page._run_button.isEnabled() is True
    assert busy_states[-1] is False

    # The completed stale bridge cannot affect the replacement run.  Its
    # release event stays set, so this second inference completes immediately.
    assert page.set_images([second])
    assert page.start_inference()
    qtbot.waitUntil(lambda: not page.is_running, timeout=3_000)  # type: ignore[attr-defined]
    assert [result.image_path for result in page.results] == [second.resolve()]
    assert len(adapter.calls) == 2


def test_each_completed_round_releases_its_signal_bridge(tmp_path: Path, qtbot: object) -> None:
    """Repeated runs retire bridges instead of leaving one child QObject per round."""
    image = _image(tmp_path / "apple.png")
    page = ImageInferencePage(model_provider=RecordingAdapter, mode="single", start_allowed=lambda: True)
    qtbot.addWidget(page)  # type: ignore[attr-defined]
    page.show()
    for _round in range(4):
        assert page.set_images([image])
        assert page.start_inference()
        qtbot.waitUntil(lambda: not page.is_running, timeout=3_000)  # type: ignore[attr-defined]
        qtbot.waitUntil(lambda: not page._bridges, timeout=3_000)  # type: ignore[attr-defined]
        assert page._signal_bridge is None


def test_export_package_failure_cleans_staging_and_never_publishes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image-render failure leaves no destination or staging package behind."""
    first = _image(tmp_path / "first.png")
    second = _image(tmp_path / "second.png")
    real_annotated_image = result_exporter._annotated_image
    calls = 0

    def fail_second(result: ImageInferenceResult, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected image renderer failure")
        real_annotated_image(result, destination)

    monkeypatch.setattr(result_exporter, "_annotated_image", fail_second)
    destination = tmp_path / "failed_export"
    with pytest.raises(ResultExportError, match="could not be completed"):
        export_inference_results([_result(first), _result(second)], destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed_export.staging.*"))
    assert not list(tmp_path.glob(".failed_export.publish.lock"))


def test_export_package_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    """A pre-existing result folder is preserved byte-for-byte."""
    image = _image(tmp_path / "apple.png")
    destination = tmp_path / "existing_export"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")
    with pytest.raises(ResultExportError, match="already exists"):
        export_inference_results([_result(image)], destination)
    assert sentinel.read_text(encoding="utf-8") == "do not replace"
    assert not list(tmp_path.glob(".existing_export.staging.*"))


def test_concurrent_exports_publish_one_complete_package_without_mixing(tmp_path: Path) -> None:
    """One same-target export wins; the other fails without altering its package."""
    image = _image(tmp_path / "apple.png")
    destination = tmp_path / "contended_export"
    barrier = threading.Barrier(2)
    successes: list[object] = []
    failures: list[BaseException] = []

    def export_once() -> None:
        try:
            barrier.wait(timeout=3)
            successes.append(export_inference_results([_result(image)], destination))
        except BaseException as error:  # Thread assertion is checked below.
            failures.append(error)

    first = threading.Thread(target=export_once)
    second = threading.Thread(target=export_once)
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert len(successes) == 1
    assert len(failures) == 1 and isinstance(failures[0], ResultExportError)
    assert (destination / "annotated_images" / "0001_apple_annotated.png").is_file()
    assert (destination / "detections.csv").is_file()
    assert (destination / "results.json").is_file()
    assert (destination / "manifest.json").is_file()
    assert not list(tmp_path.glob(".contended_export.staging.*"))
    assert not list(tmp_path.glob(".contended_export.publish.lock"))
