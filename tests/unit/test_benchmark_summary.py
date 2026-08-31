from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.cli.benchmark_model import main
import fruit_ssod.evaluation.benchmark as benchmark_module
from fruit_ssod.evaluation.benchmark import BenchmarkConfig, BenchmarkError, benchmark_model, collect_environment, file_evidence, summarize_timings, write_benchmark


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "best.pt"; path.write_bytes(b"five-fruit-final-model")
    return path


def _environment() -> dict[str, object]:
    return {
        "gpu_name": "NVIDIA GeForce RTX 3080",
        "nvidia_smi_gpu_name": "NVIDIA GeForce RTX 3080",
        "gpu_index": 0,
        "cuda_logical_index": 0,
        "nvidia_smi_physical_index": 0,
        "gpu_uuid": "GPU-3080",
        "torch_gpu_uuid": "GPU-3080",
        "driver_version": "555.85",
        "torch_version": "2.5.1+cu121",
        "cuda_runtime": "12.1",
        "ultralytics_version": "8.3.0",
        "benchmark_process_id": 4321,
        "active_compute_pids": [],
        "foreign_compute_pids": [],
        "gpu_isolation_policy": "no_foreign_compute_processes",
    }


def test_summary_uses_deterministic_linear_p95_and_is_immutable(tmp_path: Path) -> None:
    config = BenchmarkConfig(warmup_iterations=2, measured_iterations=4, image_size=640)
    summary = summarize_timings(durations_seconds=(0.010, 0.020, 0.030, 0.040), config=config, peak_allocated_bytes=123456, model=file_evidence(_model(tmp_path)), environment=_environment())
    assert summary.latency_ms["mean"] == pytest.approx(25.0)
    assert summary.latency_ms["median"] == pytest.approx(25.0)
    assert summary.latency_ms["p95"] == pytest.approx(38.5)
    assert summary.fps == pytest.approx(40.0)
    with pytest.raises(TypeError):
        summary.environment["gpu_name"] = "changed"  # type: ignore[index]


def test_summary_rejects_wrong_number_of_measurements(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="timing count"):
        summarize_timings(durations_seconds=(0.01,), config=BenchmarkConfig(measured_iterations=2), peak_allocated_bytes=0, model=file_evidence(_model(tmp_path)), environment=_environment())


def test_benchmark_requires_synchronized_explicit_cuda_ordinal() -> None:
    with pytest.raises(BenchmarkError, match="synchronization is mandatory"):
        BenchmarkConfig(synchronize=False)
    with pytest.raises(BenchmarkError, match="explicit CUDA ordinal"):
        BenchmarkConfig(device="cuda")


class _Completed:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


class _Cuda:
    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "NVIDIA GeForce RTX 3080"

    @staticmethod
    def get_device_properties(index: int) -> object:
        assert index == 0
        class _CUuuid:
            def __str__(self) -> str:
                return "3080"
        return type("Properties", (), {"uuid": _CUuuid()})()


class _Torch:
    __version__ = "2.5.1+cu121"
    cuda = _Cuda()

    class version:
        cuda = "12.1"


class _Ultralytics:
    __version__ = "8.3.0"


def test_environment_queries_same_gpu_and_records_only_own_process() -> None:
    commands: list[list[str]] = []
    def runner(command: list[str], **_kwargs: object) -> _Completed:
        commands.append(command)
        if "--query-gpu=index,name,driver_version,uuid" in command:
            return _Completed("0, NVIDIA GeForce RTX 3080, 555.85, GPU-3080\n")
        return _Completed("4321, GPU-3080\n")
    environment = collect_environment(torch_module=_Torch, ultralytics_module=_Ultralytics, device="cuda:0", command_runner=runner, process_id=4321)
    assert environment["gpu_index"] == 0
    assert environment["cuda_logical_index"] == environment["nvidia_smi_physical_index"] == 0
    assert environment["gpu_uuid"] == environment["torch_gpu_uuid"] == "GPU-3080"
    assert environment["active_compute_pids"] == [4321]
    assert environment["foreign_compute_pids"] == []
    assert len(commands) == 2


def test_environment_rejects_cuda_logical_to_physical_remapping() -> None:
    def runner(command: list[str], **_kwargs: object) -> _Completed:
        if "--query-gpu=index,name,driver_version,uuid" in command:
            return _Completed("0, NVIDIA A100, 555.85, GPU-other\n1, NVIDIA GeForce RTX 3080, 555.85, GPU-3080\n")
        return _Completed("")
    with pytest.raises(BenchmarkError, match="remapped CUDA devices are forbidden"):
        collect_environment(torch_module=_Torch, ultralytics_module=_Ultralytics, device="cuda:0", command_runner=runner, process_id=4321)


def test_environment_rejects_foreign_compute_process() -> None:
    def runner(command: list[str], **_kwargs: object) -> _Completed:
        if "--query-gpu=index,name,driver_version,uuid" in command:
            return _Completed("0, NVIDIA GeForce RTX 3080, 555.85, GPU-3080\n")
        return _Completed("4321, GPU-3080\n9999, GPU-3080\n")
    with pytest.raises(BenchmarkError, match="foreign compute PID"):
        collect_environment(torch_module=_Torch, ultralytics_module=_Ultralytics, device="cuda:0", command_runner=runner, process_id=4321)


def test_environment_rejects_rtx_3080_ti() -> None:
    class TiCuda(_Cuda):
        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "NVIDIA GeForce RTX 3080 Ti"

    class TiTorch(_Torch):
        cuda = TiCuda()

    def runner(command: list[str], **_kwargs: object) -> _Completed:
        if "--query-gpu=index,name,driver_version,uuid" in command:
            return _Completed("0, NVIDIA GeForce RTX 3080 Ti, 555.85, GPU-3080\n")
        return _Completed("")

    with pytest.raises(BenchmarkError, match="not the required RTX 3080"):
        collect_environment(torch_module=TiTorch, ultralytics_module=_Ultralytics, device="cuda:0", command_runner=runner, process_id=4321)


def test_summary_requires_direct_rtx_3080_and_isolation_evidence(tmp_path: Path) -> None:
    model = file_evidence(_model(tmp_path))

    def make_summary(environment: dict[str, object]) -> object:
        return summarize_timings(durations_seconds=(0.01,), config=BenchmarkConfig(measured_iterations=1), peak_allocated_bytes=0, model=model, environment=environment)

    incomplete = _environment()
    del incomplete["gpu_uuid"]
    with pytest.raises(BenchmarkError, match="environment metadata is incomplete"):
        make_summary(incomplete)
    foreign = _environment()
    foreign["foreign_compute_pids"] = [9999]
    with pytest.raises(BenchmarkError, match="isolation policy is violated"):
        make_summary(foreign)
    wrong_model = _environment()
    wrong_model["gpu_name"] = wrong_model["nvidia_smi_gpu_name"] = "NVIDIA GeForce RTX 3080 Ti"
    with pytest.raises(BenchmarkError, match="GPU model is incompatible"):
        make_summary(wrong_model)


def test_summary_requires_a_real_weights_path_and_consistent_mib_size(tmp_path: Path) -> None:
    model = file_evidence(_model(tmp_path))

    empty_path = dict(model); empty_path["weights_path"] = ""
    with pytest.raises(BenchmarkError, match="weights_path"):
        summarize_timings(durations_seconds=(0.01,), config=BenchmarkConfig(measured_iterations=1), peak_allocated_bytes=0, model=empty_path, environment=_environment())

    incorrect_mib = dict(model); incorrect_mib["size_mib"] = float(model["size_mib"]) + 1.0
    with pytest.raises(BenchmarkError, match="MiB"):
        summarize_timings(durations_seconds=(0.01,), config=BenchmarkConfig(measured_iterations=1), peak_allocated_bytes=0, model=incorrect_mib, environment=_environment())


def test_cli_dry_run_delays_gpu_import_and_never_writes_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    weights = _model(tmp_path); output = tmp_path / "benchmark.json"
    assert main(("--weights", str(weights), "--output", str(output), "--warmup", "1", "--iterations", "2", "--dry-run")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["authorized"] is True
    assert payload["config"]["measured_iterations"] == 2
    assert not output.exists()


def test_write_benchmark_is_non_overwriting(tmp_path: Path) -> None:
    summary = summarize_timings(durations_seconds=(0.01,), config=BenchmarkConfig(measured_iterations=1), peak_allocated_bytes=0, model=file_evidence(_model(tmp_path)), environment=_environment())
    output = write_benchmark(summary, tmp_path / "benchmark.json")
    assert json.loads(output.read_text(encoding="utf-8"))["protocol"] == "fruit_ssod_rtx3080_inference_benchmark_v1"
    with pytest.raises(BenchmarkError, match="already exists"):
        write_benchmark(summary, output)


def test_write_benchmark_cannot_overwrite_a_racing_publisher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = summarize_timings(durations_seconds=(0.01,), config=BenchmarkConfig(measured_iterations=1), peak_allocated_bytes=0, model=file_evidence(_model(tmp_path)), environment=_environment())
    output = tmp_path / "benchmark.json"
    original_link = benchmark_module.os.link
    def competing_link(source: str, destination: Path | str, *_args: object, **_kwargs: object) -> None:
        Path(destination).write_text("racing publisher", encoding="utf-8")
        original_link(source, destination)
    monkeypatch.setattr(benchmark_module.os, "link", competing_link)
    with pytest.raises(BenchmarkError, match="already exists"):
        write_benchmark(summary, output)
    assert output.read_text(encoding="utf-8") == "racing publisher"


def test_benchmark_rejects_a_checkpoint_swapped_during_private_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = _model(tmp_path)
    original_copyfile = benchmark_module.shutil.copyfile

    class SnapshotCuda:
        @staticmethod
        def is_available() -> bool:
            return True
        @staticmethod
        def get_device_properties(_device: str) -> object:
            return object()

    class SnapshotTorch:
        cuda = SnapshotCuda()

    monkeypatch.setattr(benchmark_module, "_runtime_modules", lambda: (object(), SnapshotTorch, object()))

    def copy_other_then_restore(source: str | Path, destination: str | Path, *args: object, **kwargs: object) -> str:
        weights.write_bytes(b"checkpoint-B")
        result = original_copyfile(source, destination, *args, **kwargs)
        weights.write_bytes(b"five-fruit-final-model")
        return result

    monkeypatch.setattr(benchmark_module.shutil, "copyfile", copy_other_then_restore)
    with pytest.raises(BenchmarkError, match="changed while the private snapshot was being created"):
        benchmark_model(weights=weights, config=BenchmarkConfig(warmup_iterations=1, measured_iterations=1))


def test_benchmark_rejects_foreign_pid_that_appears_during_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = _model(tmp_path)

    class RuntimeCuda:
        @staticmethod
        def is_available() -> bool:
            return True
        @staticmethod
        def get_device_properties(_device: str) -> object:
            return object()
        @staticmethod
        def synchronize(_device: str) -> None:
            return None
        @staticmethod
        def reset_peak_memory_stats(_device: str) -> None:
            return None
        @staticmethod
        def max_memory_allocated(_device: str) -> int:
            return 64

    class RuntimeTorch:
        cuda = RuntimeCuda()

    class RuntimeModel:
        names = {0: "Apple", 1: "Banana", 2: "Orange", 3: "Strawberry", 4: "Pineapple"}
        def __call__(self, *_args: object, **_kwargs: object) -> None:
            return None

    class RuntimeUltralytics:
        __version__ = "8.3.0"
        YOLO = lambda _path: RuntimeModel()

    class RuntimeNumpy:
        uint8 = object()
        @staticmethod
        def zeros(*_args: object, **_kwargs: object) -> object:
            return object()

    environments = [_environment(), {**_environment(), "active_compute_pids": [4321, 9876], "foreign_compute_pids": [9876]}]
    def collect_lifecycle_environment(**_kwargs: object) -> dict[str, object]:
        return environments.pop(0)

    monkeypatch.setattr(benchmark_module, "_runtime_modules", lambda: (RuntimeNumpy, RuntimeTorch, RuntimeUltralytics))
    with pytest.raises(BenchmarkError, match="changed during timed inference"):
        benchmark_model(weights=weights, config=BenchmarkConfig(warmup_iterations=1, measured_iterations=1), environment_collector=collect_lifecycle_environment)
