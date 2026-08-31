"""Reproducible CUDA inference benchmarking for the final fruit detector.

The benchmark deliberately measures end-to-end framework inference on a
synthetic 640x640 image.  It is not a training metric and it never imports
CUDA, PyTorch, or Ultralytics until a real benchmark is requested, which
makes the summary and CLI validation testable on a CPU-only developer host.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.detection.adapter import DetectorAdapterError, validate_class_mapping


class BenchmarkError(ValueError):
    """Raised when a deployment benchmark cannot produce honest evidence."""


def _problem(problem: str, cause: str, remediation: str) -> BenchmarkError:
    return BenchmarkError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _problem("benchmark metadata has a non-string key", "JSON would coerce the key", "use canonical string keys only")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise _problem("benchmark contains a non-finite number", "latency or metadata is NaN/infinite", "record finite measurements only")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise _problem("benchmark contains a non-JSON value", type(value).__name__, "use JSON-safe benchmark evidence")


def thaw(value: Any) -> Any:
    """Return a JSON-serializable deep copy of immutable benchmark evidence."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(thaw(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _problem(f"{name} must be a positive integer", f"received {value!r}", "supply a positive benchmark iteration count or image size")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) <= 0:
        raise _problem(f"{name} must be a finite value above zero", f"received {value!r}", "provide successful positive timing measurements")
    return float(value)


def _cuda_device_index(device: object) -> int:
    """Accept one explicit CUDA ordinal; a benchmark must not pick implicitly."""
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise _problem("benchmark device must name an explicit CUDA ordinal", f"received {device!r}", "use cuda:0 for the RTX 3080 deployment protocol")
    suffix = device.removeprefix("cuda:")
    if not suffix.isdecimal():
        raise _problem("benchmark CUDA device ordinal is malformed", f"received {device!r}", "use a non-negative explicit CUDA ordinal such as cuda:0")
    return int(suffix)


def _normalized_gpu_uuid(value: object, *, source: str) -> str:
    """Return the canonical ``GPU-...`` spelling for CUDA/SMI UUID evidence.

    PyTorch exposes ``properties.uuid`` as a private ``_CUuuid`` object in
    supported releases, whereas ``nvidia-smi`` normally prints a string with a
    ``GPU-`` prefix.  Both forms identify the same hardware, so retain one
    canonical spelling in the published record rather than comparing object
    representations or prefix variants.
    """
    if value is None:
        raise ValueError(f"{source} does not expose a usable GPU UUID")
    text = str(value).strip()
    if text.casefold().startswith("gpu-"):
        text = text[4:].strip()
    if not text:
        raise ValueError(f"{source} does not expose a usable GPU UUID")
    return f"GPU-{text}"


def _is_required_rtx_3080(name: str) -> bool:
    """Accept exactly the desktop RTX 3080 model, never a Ti/laptop variant."""
    normalized = " ".join(name.casefold().split())
    return normalized in {"rtx 3080", "nvidia rtx 3080", "nvidia geforce rtx 3080"}


@dataclass(frozen=True)
class BenchmarkConfig:
    """The fixed protocol fields that define a comparable deployment run."""

    warmup_iterations: int = 20
    measured_iterations: int = 100
    image_size: int = 640
    device: str = "cuda:0"
    synchronize: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "warmup_iterations", _positive_int(self.warmup_iterations, "warmup_iterations"))
        object.__setattr__(self, "measured_iterations", _positive_int(self.measured_iterations, "measured_iterations"))
        object.__setattr__(self, "image_size", _positive_int(self.image_size, "image_size"))
        _cuda_device_index(self.device)
        if self.synchronize is not True:
            raise _problem("CUDA synchronization is mandatory", f"received synchronize={self.synchronize!r}", "keep synchronization enabled before and after every measured inference")

    def mapping(self) -> dict[str, Any]:
        return {"warmup_iterations": self.warmup_iterations, "measured_iterations": self.measured_iterations, "image_size": self.image_size, "device": self.device, "synchronize": self.synchronize}


@dataclass(frozen=True)
class BenchmarkSummary:
    """Portable, deeply immutable benchmark result and provenance envelope."""

    config: BenchmarkConfig
    latency_ms: Mapping[str, float]
    fps: float
    peak_allocated_bytes: int
    model: Mapping[str, Any]
    environment: Mapping[str, Any]
    schema_version: str = "1.0"
    protocol: str = "fruit_ssod_rtx3080_inference_benchmark_v1"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or self.protocol != "fruit_ssod_rtx3080_inference_benchmark_v1":
            raise _problem("benchmark schema or protocol is unsupported", "the result cannot be compared under the approved deployment protocol", "use the current benchmark implementation")
        expected = {"mean", "median", "p95", "min", "max"}
        if not isinstance(self.latency_ms, Mapping) or set(self.latency_ms) != expected:
            raise _problem("latency summary is incomplete", "required latency statistics are missing", "record mean, median, p95, min, and max latency in milliseconds")
        frozen_latency = _freeze(dict(self.latency_ms))
        assert isinstance(frozen_latency, Mapping)
        for name, value in frozen_latency.items():
            _finite_positive(value, f"latency_ms.{name}")
        mean_latency = float(frozen_latency["mean"])
        if not math.isclose(float(self.fps), 1000.0 / mean_latency, rel_tol=1e-9, abs_tol=1e-9):
            raise _problem("FPS does not match mean latency", "benchmark evidence is internally inconsistent", "compute FPS as 1000 divided by measured mean latency")
        object.__setattr__(self, "fps", _finite_positive(self.fps, "fps"))
        if isinstance(self.peak_allocated_bytes, bool) or not isinstance(self.peak_allocated_bytes, int) or self.peak_allocated_bytes < 0:
            raise _problem("peak allocated memory is invalid", f"received {self.peak_allocated_bytes!r}", "record a non-negative integer byte count from torch.cuda.max_memory_allocated")
        frozen_model = _freeze(self.model)
        frozen_environment = _freeze(self.environment)
        if not isinstance(frozen_model, Mapping) or not isinstance(frozen_environment, Mapping):
            raise _problem("benchmark model/environment is malformed", "top-level evidence must be JSON objects", "record model and environment metadata objects")
        required_model = {"weights_path", "weights_sha256", "size_bytes", "size_mib"}
        if not required_model.issubset(frozen_model):
            raise _problem("model evidence is incomplete", "weights identity or size is absent", "record weight path, SHA-256, bytes, and MiB")
        if not isinstance(frozen_model["weights_path"], str) or not frozen_model["weights_path"].strip():
            raise _problem("model weights_path is invalid", "the audited checkpoint path is empty or not text", "record the nonempty resolved path supplied for the final checkpoint")
        digest = frozen_model["weights_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise _problem("model SHA-256 is malformed", "weights evidence cannot identify the model", "record the 64-character SHA-256 of the benchmarked weights")
        if isinstance(frozen_model["size_bytes"], bool) or not isinstance(frozen_model["size_bytes"], int) or frozen_model["size_bytes"] <= 0:
            raise _problem("model size is invalid", "weights evidence is empty or malformed", "benchmark a nonempty model weight file")
        size_mib = _finite_positive(frozen_model["size_mib"], "model.size_mib")
        expected_mib = frozen_model["size_bytes"] / (1024 * 1024)
        if not math.isclose(size_mib, expected_mib, rel_tol=1e-9, abs_tol=1e-9):
            raise _problem("model size MiB is inconsistent", "size_mib does not match size_bytes divided by 1024 squared", "compute MiB directly from the recorded checkpoint byte count")
        required_environment = {
            "gpu_name", "nvidia_smi_gpu_name", "gpu_index", "cuda_logical_index",
            "nvidia_smi_physical_index", "gpu_uuid", "torch_gpu_uuid", "driver_version",
            "torch_version", "cuda_runtime", "ultralytics_version", "benchmark_process_id",
            "active_compute_pids", "foreign_compute_pids", "gpu_isolation_policy",
        }
        missing = sorted(required_environment.difference(frozen_environment))
        string_environment = {
            "gpu_name", "nvidia_smi_gpu_name", "gpu_uuid", "torch_gpu_uuid", "driver_version",
            "torch_version", "cuda_runtime", "ultralytics_version", "gpu_isolation_policy",
        }
        if missing or any(not isinstance(frozen_environment[key], str) or not frozen_environment[key].strip() for key in string_environment):
            raise _problem("benchmark environment metadata is incomplete", f"missing or empty fields: {missing or 'one or more required values'}", "collect GPU, driver, PyTorch, CUDA, and Ultralytics versions before publishing")
        gpu_name = frozen_environment["gpu_name"]
        smi_name = frozen_environment["nvidia_smi_gpu_name"]
        if " ".join(gpu_name.casefold().split()) != " ".join(smi_name.casefold().split()) or not _is_required_rtx_3080(gpu_name):
            raise _problem("benchmark GPU model is incompatible", "PyTorch and nvidia-smi must agree on exactly NVIDIA GeForce RTX 3080", "run the approved benchmark on the RTX 3080, not a Ti, laptop, or other GPU")
        try:
            recorded_gpu_uuid = _normalized_gpu_uuid(frozen_environment["gpu_uuid"], source="benchmark GPU UUID")
            recorded_torch_uuid = _normalized_gpu_uuid(frozen_environment["torch_gpu_uuid"], source="benchmark PyTorch GPU UUID")
        except ValueError as error:
            raise _problem("benchmark GPU UUID is malformed", str(error), "record matching canonical GPU UUID evidence") from error
        if (recorded_gpu_uuid != frozen_environment["gpu_uuid"] or recorded_torch_uuid != frozen_environment["torch_gpu_uuid"] or recorded_gpu_uuid.casefold() != recorded_torch_uuid.casefold()):
            raise _problem("benchmark GPU UUID evidence is inconsistent", "PyTorch and nvidia-smi UUIDs must be matching canonical GPU-prefixed values", "collect the environment with the approved benchmark command")
        for key in ("gpu_index", "cuda_logical_index", "nvidia_smi_physical_index", "benchmark_process_id"):
            value = frozen_environment[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (key == "benchmark_process_id" and value == 0):
                raise _problem("benchmark environment index or process ID is invalid", f"{key}={value!r}", "record non-negative GPU indices and a positive benchmark process ID")
        if not (frozen_environment["gpu_index"] == frozen_environment["cuda_logical_index"] == frozen_environment["nvidia_smi_physical_index"]):
            raise _problem("benchmark CUDA-to-SMI GPU mapping is inconsistent", "logical and physical indices differ", "reject CUDA device remapping and benchmark the direct RTX 3080 ordinal")
        for key in ("active_compute_pids", "foreign_compute_pids"):
            pids = frozen_environment[key]
            if not isinstance(pids, tuple) or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids) or tuple(sorted(set(pids))) != pids:
                raise _problem("benchmark GPU process evidence is invalid", f"{key} must be sorted unique positive process IDs", "collect nvidia-smi process evidence with the approved benchmark command")
        if frozen_environment["gpu_isolation_policy"] != "no_foreign_compute_processes" or frozen_environment["foreign_compute_pids"]:
            raise _problem("benchmark GPU isolation policy is violated", "the RTX 3080 has foreign compute processes or an unapproved policy", "close foreign GPU compute workloads before benchmarking")
        object.__setattr__(self, "latency_ms", frozen_latency)
        object.__setattr__(self, "model", frozen_model)
        object.__setattr__(self, "environment", frozen_environment)

    def mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "config": self.config.mapping(),
            "latency_ms": thaw(self.latency_ms),
            "fps": self.fps,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_allocated_mib": self.peak_allocated_bytes / (1024 * 1024),
            "model": thaw(self.model),
            "environment": thaw(self.environment),
        }


def summarize_timings(*, durations_seconds: Iterable[float], config: BenchmarkConfig, peak_allocated_bytes: int, model: Mapping[str, Any], environment: Mapping[str, Any]) -> BenchmarkSummary:
    """Build a deterministic summary from already synchronized measured calls.

    This pure function is used by unit tests and keeps quantile definition
    explicit: p95 is linear interpolation over sorted samples.
    """
    samples = sorted(_finite_positive(item, "measured duration") for item in durations_seconds)
    if len(samples) != config.measured_iterations:
        raise _problem("measured timing count differs from the protocol", f"expected {config.measured_iterations}, got {len(samples)}", "record exactly the configured number of measured inference calls")
    def percentile(values: Sequence[float], fraction: float) -> float:
        position = (len(values) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return values[low]
        return values[low] + (values[high] - values[low]) * (position - low)
    latency = {
        "mean": statistics.fmean(samples) * 1000.0,
        "median": statistics.median(samples) * 1000.0,
        "p95": percentile(samples, 0.95) * 1000.0,
        "min": samples[0] * 1000.0,
        "max": samples[-1] * 1000.0,
    }
    return BenchmarkSummary(config=config, latency_ms=latency, fps=1000.0 / latency["mean"], peak_allocated_bytes=peak_allocated_bytes, model=model, environment=environment)


def file_evidence(weights: Path | str) -> dict[str, Any]:
    """Return model identity evidence, refusing symlinks and empty files."""
    path = Path(weights)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise _problem("benchmark weights cannot be resolved", str(error), "pass a readable final best.pt file") from error
    if path.is_symlink() or not resolved.is_file():
        raise _problem("benchmark weights are not a regular file", str(path), "pass the released non-symlink best.pt file")
    try:
        content = resolved.read_bytes()
    except OSError as error:
        raise _problem("benchmark weights cannot be read", str(error), "restore read access to the final checkpoint") from error
    if not content:
        raise _problem("benchmark weights are empty", str(resolved), "pass the nonempty final best.pt checkpoint")
    return {"weights_path": str(resolved), "weights_sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content), "size_mib": len(content) / (1024 * 1024)}


def _same_weight_content(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare checkpoint identity while allowing the two paths to differ."""
    return (
        left.get("weights_sha256") == right.get("weights_sha256")
        and left.get("size_bytes") == right.get("size_bytes")
        and left.get("size_mib") == right.get("size_mib")
    )


def _create_private_weight_snapshot(weights: Path | str, directory: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Copy and seal a checkpoint before model construction.

    The model must never load the caller-controlled checkpoint directly: a
    file may be replaced between a pre-load digest and ``YOLO(...)``.  We hash
    the source both before and after copying and compare the private copy, so
    a transient replacement (including B->A restoration) is rejected.
    """
    source = file_evidence(weights)
    snapshot_path = directory / "loaded_checkpoint.pt"
    try:
        shutil.copyfile(source["weights_path"], snapshot_path)
    except OSError as error:
        raise _problem("benchmark weights cannot be privately snapshotted", str(error), "ensure the final checkpoint is readable and the temporary directory is writable") from error
    snapshot = file_evidence(snapshot_path)
    settled_source = file_evidence(weights)
    if not _same_weight_content(source, settled_source) or not _same_weight_content(source, snapshot):
        raise _problem("benchmark weights changed while the private snapshot was being created", "the source checkpoint and private copy do not have one stable SHA-256 and size", "stop checkpoint publication, restore the final immutable best.pt, and rerun the benchmark")
    return source, snapshot, snapshot_path


def _published_model_evidence(*, source: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Publish actual loaded-content evidence while retaining the input path."""
    return {
        # These canonical identity fields describe exactly the temporary file
        # given to YOLO.  The source path below keeps the final-result record
        # traceable after the private temporary directory is removed.
        "weights_path": str(source["weights_path"]),
        "weights_sha256": str(snapshot["weights_sha256"]),
        "size_bytes": int(snapshot["size_bytes"]),
        "size_mib": float(snapshot["size_mib"]),
        "source_weights_path": str(source["weights_path"]),
        "source_weights_sha256": str(source["weights_sha256"]),
        "source_size_bytes": int(source["size_bytes"]),
        "loaded_snapshot_sha256": str(snapshot["weights_sha256"]),
        "loaded_snapshot_size_bytes": int(snapshot["size_bytes"]),
    }


def _verify_environment_stable(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    """Reject environmental drift or a new competing process during timing."""
    static_fields = (
        "gpu_name", "nvidia_smi_gpu_name", "gpu_index", "cuda_logical_index",
        "nvidia_smi_physical_index", "gpu_uuid", "torch_gpu_uuid", "driver_version",
        "torch_version", "cuda_runtime", "ultralytics_version", "benchmark_process_id",
        "gpu_isolation_policy",
    )
    changed = [field for field in static_fields if before.get(field) != after.get(field)]
    if changed:
        raise _problem("GPU environment changed during timed inference", f"the final probe differs for {changed!r}", "keep the same direct RTX 3080 device and driver/runtime environment for the entire benchmark")
    own_pid = after.get("benchmark_process_id")
    after_active = after.get("active_compute_pids")
    after_foreign = after.get("foreign_compute_pids")
    if not isinstance(own_pid, int) or isinstance(own_pid, bool) or own_pid <= 0 or not isinstance(after_active, list) or not isinstance(after_foreign, list):
        raise _problem("GPU environment changed during timed inference", "the post-timing process evidence is malformed", "collect a complete final nvidia-smi GPU process probe")
    observed_foreign = sorted(set(after_active).difference({own_pid}))
    if observed_foreign or after_foreign:
        raise _problem("GPU environment changed during timed inference", f"new or existing foreign compute PID(s) were observed: {sorted(set(observed_foreign).union(after_foreign))!r}", "close all foreign RTX 3080 compute workloads and rerun the complete benchmark")


def collect_environment(*, torch_module: Any, ultralytics_module: Any, device: str, command_runner: Callable[..., Any] = subprocess.run, process_id: int | None = None) -> dict[str, Any]:
    """Collect one selected RTX 3080's versions and isolation evidence.

    CUDA logical ordinals can be remapped by CUDA_VISIBLE_DEVICES.  Therefore
    the logical ordinal's PyTorch UUID is first matched against every physical
    ``nvidia-smi`` GPU row, and the run is rejected unless both ordinals agree.
    The selected card must have no foreign compute PID. The benchmark process
    itself is allowed because CUDA initialization often makes it appear in the
    process table before timing starts.
    """
    index = _cuda_device_index(device)
    own_pid = os.getpid() if process_id is None else process_id
    if isinstance(own_pid, bool) or not isinstance(own_pid, int) or own_pid <= 0:
        raise _problem("benchmark process identifier is invalid", f"received {own_pid!r}", "run the benchmark from a normal operating-system process")
    try:
        gpu_name = str(torch_module.cuda.get_device_name(index))
        properties = torch_module.cuda.get_device_properties(index)
        torch_uuid = _normalized_gpu_uuid(getattr(properties, "uuid", None), source="PyTorch CUDA device properties")
        cuda_runtime = str(torch_module.version.cuda)
        torch_version = str(torch_module.__version__)
        ultralytics_version = str(ultralytics_module.__version__)
        selected = command_runner(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        lines = [line.strip() for line in str(selected.stdout).splitlines() if line.strip()]
        physical_rows: list[tuple[int, str, str, str]] = []
        for line in lines:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4 or not fields[0].isdecimal() or not all(fields[1:]):
                raise ValueError("nvidia-smi GPU row does not contain valid index, name, driver, and UUID")
            physical_rows.append((int(fields[0]), fields[1], fields[2], _normalized_gpu_uuid(fields[3], source="nvidia-smi GPU row")))
        matches = [row for row in physical_rows if row[3].casefold() == torch_uuid.casefold()]
        if len(matches) != 1:
            raise ValueError(f"PyTorch GPU UUID {torch_uuid!r} matches {len(matches)} nvidia-smi GPU rows")
        physical_index, smi_gpu, driver, gpu_uuid = matches[0]
        if physical_index != index:
            raise ValueError(f"CUDA logical device {index} maps via UUID {torch_uuid!r} to physical NVIDIA GPU {physical_index}; remapped CUDA devices are forbidden for this benchmark")
        normalized_torch_name = " ".join(gpu_name.casefold().split())
        normalized_smi_name = " ".join(smi_gpu.casefold().split())
        if normalized_torch_name != normalized_smi_name:
            raise ValueError(f"PyTorch selected {gpu_name!r}, but nvidia-smi UUID-matched GPU {physical_index} reports {smi_gpu!r}")
        if not _is_required_rtx_3080(gpu_name):
            raise ValueError(f"selected GPU {gpu_name!r} is not the required RTX 3080")
        if not gpu_name or not cuda_runtime or not torch_version or not ultralytics_version or not smi_gpu or not driver or not gpu_uuid:
            raise ValueError("one or more version fields is empty")
        process_query = command_runner(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        active_pids: list[int] = []
        for line in (item.strip() for item in str(process_query.stdout).splitlines()):
            if not line or line.casefold().startswith("no running processes"):
                continue
            pid_text, separator, row_uuid = line.partition(",")
            if not separator or not pid_text.strip().isdecimal() or not row_uuid.strip():
                raise ValueError(f"nvidia-smi compute-process row is malformed: {line!r}")
            if _normalized_gpu_uuid(row_uuid, source="nvidia-smi compute-process row").casefold() == gpu_uuid.casefold():
                active_pids.append(int(pid_text.strip()))
        active_pids = sorted(set(active_pids))
        foreign_pids = [pid for pid in active_pids if pid != own_pid]
        if foreign_pids:
            raise ValueError(f"selected GPU index {index} has foreign compute PID(s) {foreign_pids!r}")
    except (AttributeError, IndexError, OSError, subprocess.SubprocessError, ValueError) as error:
        raise _problem("GPU environment metadata or isolation cannot be verified", str(error), "ensure nvidia-smi is available, select the RTX 3080, and close other GPU compute workloads") from error
    return {
        "gpu_name": gpu_name,
        "nvidia_smi_gpu_name": smi_gpu,
        "gpu_index": index,
        "cuda_logical_index": index,
        "nvidia_smi_physical_index": physical_index,
        "gpu_uuid": gpu_uuid,
        "torch_gpu_uuid": torch_uuid,
        "driver_version": driver,
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
        "ultralytics_version": ultralytics_version,
        "benchmark_process_id": own_pid,
        "active_compute_pids": active_pids,
        "foreign_compute_pids": foreign_pids,
        "gpu_isolation_policy": "no_foreign_compute_processes",
    }


def _runtime_modules() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        import ultralytics
    except ModuleNotFoundError as error:
        raise _problem("benchmark dependency is unavailable", f"{error.name!r} is not installed", "activate the fruit-ssod Conda environment and install the locked dependencies") from error
    return np, torch, ultralytics


def benchmark_model(
    *,
    weights: Path | str,
    config: BenchmarkConfig = BenchmarkConfig(),
    environment_collector: Callable[..., dict[str, Any]] = collect_environment,
) -> BenchmarkSummary:
    """Run one synchronized CUDA benchmark of a five-class Ultralytics model."""
    np, torch, ultralytics = _runtime_modules()
    if not bool(torch.cuda.is_available()):
        raise _problem("CUDA is unavailable", "PyTorch cannot access an NVIDIA GPU", "activate the CUDA-enabled fruit-ssod Conda environment and verify nvidia-smi")
    try:
        torch.cuda.get_device_properties(config.device)
    except Exception as error:
        raise _problem("requested CUDA device is unavailable", str(error), "use cuda:0 for the RTX 3080 or select an available CUDA device") from error
    with tempfile.TemporaryDirectory(prefix="fruit_ssod_benchmark_") as private_directory:
        source_evidence, snapshot_evidence, snapshot_path = _create_private_weight_snapshot(weights, Path(private_directory))
        try:
            model = ultralytics.YOLO(str(snapshot_path))
            validate_class_mapping(getattr(model, "names", None), DEFAULT_CLASS_REGISTRY)
        except DetectorAdapterError as error:
            raise _problem("benchmark model class mapping is incompatible", str(error), "benchmark only the canonical five-class fruit detector") from error
        except Exception as error:
            raise _problem("benchmark model could not be loaded", str(error), "verify that the final checkpoint is a readable Ultralytics model") from error
        loaded_snapshot = file_evidence(snapshot_path)
        if not _same_weight_content(snapshot_evidence, loaded_snapshot):
            raise _problem("benchmark weights changed while the private snapshot was loading", "the actual YOLO input no longer matches the sealed private checkpoint SHA-256 and size", "rerun from a trusted local environment after checking for checkpoint tampering")
        model_evidence = _published_model_evidence(source=source_evidence, snapshot=loaded_snapshot)
        environment = environment_collector(torch_module=torch, ultralytics_module=ultralytics, device=config.device)
        image = np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8)
        def infer() -> None:
            model(image, imgsz=config.image_size, device=config.device, verbose=False)
        def synchronize() -> None:
            torch.cuda.synchronize(config.device)
        try:
            for _ in range(config.warmup_iterations):
                infer()
            synchronize()
            torch.cuda.reset_peak_memory_stats(config.device)
            durations: list[float] = []
            for _ in range(config.measured_iterations):
                synchronize()
                start = time.perf_counter()
                infer()
                synchronize()
                durations.append(time.perf_counter() - start)
            # A final explicit synchronization closes the measured region
            # before the independent post-timing GPU-process/environment probe.
            synchronize()
            peak = int(torch.cuda.max_memory_allocated(config.device))
        except Exception as error:
            raise _problem("CUDA inference benchmark failed", str(error), "close other GPU workloads, confirm the checkpoint and reduce only after recording an OOM failure") from error
        final_environment = environment_collector(torch_module=torch, ultralytics_module=ultralytics, device=config.device)
        _verify_environment_stable(environment, final_environment)
        return summarize_timings(durations_seconds=durations, config=config, peak_allocated_bytes=peak, model=model_evidence, environment=final_environment)


def write_benchmark(summary: BenchmarkSummary, path: Path | str) -> Path:
    """Atomically write a new benchmark result without overwriting evidence."""
    destination = Path(path)
    if destination.exists():
        raise _problem("benchmark output already exists", str(destination), "preserve the immutable benchmark or choose a new output path")
    temporary: str | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(json.dumps(summary.mapping(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        # ``os.replace`` would reintroduce a TOCTOU overwrite between the
        # existence check and publication.  A hard-link create is exclusive
        # and atomic on the same filesystem (the temp file is in destination's
        # parent), so exactly one concurrent publisher can win.
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise _problem("benchmark output already exists", str(destination), "preserve the immutable benchmark or choose a new output path") from error
        return destination
    except OSError as error:
        raise _problem("benchmark output could not be published", str(error), "ensure the output directory is writable and retry") from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
