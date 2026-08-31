"""Thread-safe single-model ownership for the desktop GUI.

Checkpoint construction is deliberately kept out of the Qt GUI thread.  An
Ultralytics load can import PyTorch, deserialize a checkpoint, and allocate CUDA
memory; doing that from a button handler makes the window appear frozen.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from fruit_ssod.detection import DetectorAdapter, DetectorAdapterError, UltralyticsDetectorAdapter


class ModelManagerError(RuntimeError):
    """An actionable model-selection failure suitable for direct display in the GUI."""


AdapterFactory = Callable[..., DetectorAdapter]


def _error(problem: str, cause: str, remediation: str) -> ModelManagerError:
    return ModelManagerError(
        f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."
    )


def _empty_cuda_cache() -> None:
    """Best-effort cleanup without making PyTorch a GUI import-time dependency."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Optional cleanup must never prevent a user from replacing or releasing a model.
        return


def _initialize_and_validate(adapter: DetectorAdapter) -> None:
    """Run the adapter's lazy loader and canonical class-mapping validation."""
    initializer = getattr(adapter, "initialize", None)
    if not callable(initializer):
        raise _error(
            "detector adapter does not provide checkpoint compatibility validation",
            f"received {type(adapter).__name__}",
            "use the bundled Ultralytics detector adapter for GUI model loading",
        )
    initializer()


class ModelLoadWorker(QObject):
    """Construct one non-Qt detector adapter in a dedicated loading thread."""

    loaded = Signal(object, str, int)
    failed = Signal(str, int)
    finished = Signal()

    def __init__(
        self, *, weights_path: Path, adapter_factory: AdapterFactory, generation: int
    ) -> None:
        super().__init__()
        self._weights_path = weights_path
        self._adapter_factory = adapter_factory
        self._generation = generation

    @Slot()
    def load(self) -> None:
        """Load and validate, returning ownership only through a queued Qt signal."""
        adapter: DetectorAdapter | None = None
        try:
            adapter = self._adapter_factory(weights_path=self._weights_path)
            if isinstance(adapter, QObject):
                raise _error(
                    "detector adapter has unsafe Qt thread ownership",
                    f"received QObject-based {type(adapter).__name__}",
                    "use a backend-neutral DetectorAdapter that does not inherit QObject",
                )
            _initialize_and_validate(adapter)
            if QThread.currentThread().isInterruptionRequested():
                raise _error(
                    "model loading was cancelled",
                    "the application is closing or another shutdown was requested",
                    "wait for shutdown to finish, then reopen the application before loading a model",
                )
            # The adapter is a non-QObject backend value.  The queued signal transfers
            # Python ownership to ModelManager's GUI thread without moving a QObject.
            self.loaded.emit(adapter, str(self._weights_path), self._generation)
            adapter = None
        except ModelManagerError as error:
            self.failed.emit(str(error), self._generation)
        except DetectorAdapterError as error:
            self.failed.emit(
                str(
                    _error(
                        "model checkpoint is incompatible",
                        str(error),
                        "use a readable .pt checkpoint with IDs 0-4 mapped to the five canonical fruit names",
                    )
                ),
                self._generation,
            )
        except Exception as error:
            self.failed.emit(
                str(
                    _error(
                        "model checkpoint could not be loaded",
                        str(error),
                        "verify the checkpoint file, Ultralytics installation, and available GPU/CPU memory",
                    )
                ),
                self._generation,
            )
        finally:
            # Failed or cancelled loads must not retain a temporary GPU model in the
            # worker after its event loop exits.
            if adapter is not None:
                del adapter
                gc.collect()
                _empty_cuda_cache()
            self.finished.emit()


class ModelManager(QObject):
    """Own at most one detector adapter while loading checkpoints off the GUI thread.

    The manager intentionally accepts model files only.  It neither opens a camera
    nor exposes the deferred open-world extension.  A failed replacement leaves no
    stale active model: the prior adapter is released before the worker allocates the
    replacement, so two GPU-resident detectors never coexist.
    """

    model_loading = Signal(str)
    model_loaded = Signal(str)
    model_released = Signal()
    loading_finished = Signal(bool)
    status_changed = Signal(str)
    load_failed = Signal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        adapter_factory: AdapterFactory = UltralyticsDetectorAdapter,
    ) -> None:
        super().__init__(parent)
        self._adapter_factory = adapter_factory
        self._active_adapter: DetectorAdapter | None = None
        self._weights_path: Path | None = None
        self._load_thread: QThread | None = None
        self._load_worker: ModelLoadWorker | None = None
        # A worker may enqueue its success signal just before shutdown blocks the
        # GUI thread in ``QThread.wait``.  Generations make that queued payload
        # unambiguously stale even if a new model is selected before the GUI event
        # loop gets a chance to deliver it.
        self._next_load_generation = 0
        self._active_load_generation: int | None = None
        self._cancelled_load_generations: set[int] = set()

    @property
    def active_model(self) -> DetectorAdapter | None:
        """Return the only active detector, or ``None`` when no compatible model is loaded."""
        return self._active_adapter

    @property
    def active_weights_path(self) -> Path | None:
        """Return the selected weights path, if an adapter is active."""
        return self._weights_path

    @property
    def has_active_model(self) -> bool:
        """Whether a later file-inference workflow may use the loaded detector."""
        return self._active_adapter is not None

    @property
    def is_loading(self) -> bool:
        """Whether a checkpoint worker is still alive and owns the loading transition."""
        return self._load_thread is not None

    def start_loading(self, weights_path: str | Path) -> bool:
        """Start asynchronous checkpoint validation and return immediately to the GUI.

        Invalid paths fail synchronously because no worker or GPU resource is needed.
        All checkpoint construction and compatibility checks happen in
        :class:`ModelLoadWorker` on a ``QThread``.
        """
        try:
            path = self._validate_weights_path(weights_path)
        except ModelManagerError as error:
            self._report_failure(error)
            return False
        if self.is_loading:
            self._report_failure(
                _error(
                    "another model is still loading",
                    "only one checkpoint load may own GPU resources at a time",
                    "wait for the current load to finish before selecting another .pt file",
                )
            )
            return False

        # Preserve the one-active-model policy before the worker can construct a new
        # backend.  This intentionally means a failed replacement leaves no old model
        # behind, rather than silently retaining a potentially unintended checkpoint.
        self.release_model(emit_status=False)
        self._start_worker(path)
        self.model_loading.emit(path.name)
        self.status_changed.emit(f"Loading and validating model: {path.name}")
        return True

    def release_model(self, *, emit_status: bool = True) -> None:
        """Drop the sole active model reference and request CUDA cache cleanup."""
        had_model = self._active_adapter is not None
        self._active_adapter = None
        self._weights_path = None
        gc.collect()
        _empty_cuda_cache()
        if had_model:
            if emit_status:
                self.status_changed.emit("Active model released; GPU cache cleared when available.")
            self.model_released.emit()

    def shutdown(self, *, wait_ms: int = 5_000) -> bool:
        """Release the active model and safely stop a loading thread before window close.

        ``False`` means the worker did not stop in time.  Callers must keep the window
        alive in that case; destroying a running ``QThread`` is unsafe.
        """
        self.release_model(emit_status=False)
        thread = self._load_thread
        if thread is None:
            return True
        generation = self._active_load_generation
        if generation is not None:
            self._cancelled_load_generations.add(generation)
        thread.requestInterruption()
        thread.quit()
        if not thread.wait(wait_ms):
            self._report_failure(
                _error(
                    "model-loading worker is still shutting down",
                    "checkpoint initialization has not returned yet",
                    "wait a moment and close the window again; do not force-terminate the application",
                )
            )
            return False
        # ``wait`` deliberately blocks the GUI event loop, so a previously queued
        # ``loaded`` signal can only arrive after this method returns.  Clear the
        # finished thread references now; the generation cancellation above makes
        # the late success payload release its adapter instead of reactivating it.
        self._on_worker_thread_finished(thread)
        return True

    @staticmethod
    def _validate_weights_path(weights_path: str | Path) -> Path:
        path = Path(weights_path).expanduser()
        if not path.is_file():
            raise _error(
                "model weights file was not found",
                f"{path} is not an existing file",
                "select the trained best.pt checkpoint produced by this project",
            )
        if path.suffix.lower() != ".pt":
            raise _error(
                "model weights file has an unsupported format",
                f"expected a .pt checkpoint but received {path.suffix or 'no extension'}",
                "select a compatible Ultralytics .pt checkpoint trained for the five fruit classes",
            )
        return path.resolve()

    def _start_worker(self, path: Path) -> None:
        self._next_load_generation += 1
        generation = self._next_load_generation
        self._active_load_generation = generation
        thread = QThread(self)
        worker = ModelLoadWorker(
            weights_path=path,
            adapter_factory=self._adapter_factory,
            generation=generation,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.load)
        worker.loaded.connect(self._on_worker_loaded)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_worker_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    @Slot(object, str, int)
    def _on_worker_loaded(self, adapter: object, weights_path: str, generation: int) -> None:
        if (
            generation != self._active_load_generation
            or generation in self._cancelled_load_generations
        ):
            # The adapter can be held solely by this queued signal after shutdown.
            # Explicitly drop it here rather than allowing a stale GPU model to
            # survive until an arbitrary later GC cycle.
            del adapter
            gc.collect()
            _empty_cuda_cache()
            return
        if not isinstance(adapter, DetectorAdapter):
            self._on_worker_failed(
                str(
                    _error(
                        "model worker returned an invalid detector adapter",
                        f"received {type(adapter).__name__}",
                        "use the bundled DetectorAdapter implementation for GUI model loading",
                    )
                ),
                generation,
            )
            return
        self._active_adapter = adapter
        self._weights_path = Path(weights_path)
        model_name = self._weights_path.name if self._weights_path is not None else "selected model"
        self.status_changed.emit(f"Loaded active model: {model_name}")
        self.model_loaded.emit(model_name)

    @Slot(str, int)
    def _on_worker_failed(self, message: str, generation: int) -> None:
        if (
            generation != self._active_load_generation
            or generation in self._cancelled_load_generations
        ):
            return
        self._report_failure(ModelManagerError(message))

    @Slot()
    def _on_worker_thread_finished(self, finished_thread: QThread | None = None) -> None:
        if finished_thread is None:
            sender = self.sender()
            if not isinstance(sender, QThread):
                return
            finished_thread = sender
        if finished_thread is not None and finished_thread is not self._load_thread:
            return
        self._load_thread = None
        self._load_worker = None
        self.loading_finished.emit(self.has_active_model)

    def _report_failure(self, error: ModelManagerError) -> None:
        message = str(error)
        self.status_changed.emit(message)
        self.load_failed.emit(message)
