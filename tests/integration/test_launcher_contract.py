"""Static launcher contracts: no test starts Qt, Conda training, or GPU work."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
START_GUI = REPOSITORY_ROOT / "scripts" / "start_gui.ps1"
RUN_PIPELINE = REPOSITORY_ROOT / "scripts" / "run_pipeline.ps1"
README = REPOSITORY_ROOT / "README.md"
USER_GUIDE = REPOSITORY_ROOT / "docs" / "user-guide.md"
TROUBLESHOOTING = REPOSITORY_ROOT / "docs" / "troubleshooting.md"
RUN_ALL_CHECKS = REPOSITORY_ROOT / "scripts" / "run_all_checks.ps1"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_gui_launcher_resolves_repo_runs_conda_and_preflights_before_qt() -> None:
    text = _text(START_GUI)

    assert "$env:PYTHONNOUSERSITE = '1'" in text
    assert "Resolve-Path (Join-Path $PSScriptRoot '..')" in text
    assert "'fruit-ssod'" in text
    assert "FRUIT_SSOD_CONDA_ENV" in text
    assert "preflight.ps1" in text
    assert "resolve_conda_command.ps1" in text
    assert "Resolve-CondaCommand -CondaExecutable $CondaExecutable" in text
    assert "@{\n    PythonExecutable" in text
    assert "-PreflightOnly" in text
    assert "python', '-m', 'fruit_ssod.gui.app'" in text
    assert text.index("& $PreflightScript") < text.index("'fruit_ssod.gui.app'")
    assert "camera" in text.lower()
    assert "open-world" in text.lower()


def test_pipeline_launcher_preflights_and_delegates_to_controlled_matrices() -> None:
    text = _text(RUN_PIPELINE)

    assert "$env:PYTHONNOUSERSITE = '1'" in text
    for required in (
        "Resolve-Path (Join-Path $PSScriptRoot '..')",
        "run_supervised_matrix.ps1",
        "run_ssod_matrix.ps1",
        "FRUIT_SSOD_PRETRAINED_WEIGHTS",
        "FRUIT_SSOD_DATA_ROOT",
        "FRUIT_SSOD_ARTIFACT_ROOT",
        "-DryRun",
        "-Resume",
        "-Device",
        "resolve_conda_command.ps1",
        "Resolve-CondaCommand -CondaExecutable $CondaExecutable",
    ):
        assert required in text
    assert text.index("& $PreflightScript") < text.index("Invoke-Supervised")
    assert text.index("& $PreflightScript") < text.index("Invoke-Ssod")


def test_all_conda_launchers_use_the_shared_scalar_resolver() -> None:
    helper = REPOSITORY_ROOT / "scripts" / "resolve_conda_command.ps1"
    helper_text = _text(helper)
    assert "return [string]$selected" in helper_text
    assert "'conda.exe'" in helper_text and "'conda.bat'" in helper_text
    for launcher_name in (
        "start_gui.ps1",
        "run_pipeline.ps1",
        "run_supervised_matrix.ps1",
        "run_ssod_matrix.ps1",
    ):
        text = _text(REPOSITORY_ROOT / "scripts" / launcher_name)
        assert "resolve_conda_command.ps1" in text
        assert "Resolve-CondaCommand -CondaExecutable $CondaExecutable" in text
        assert "$env:PYTHONNOUSERSITE = '1'" in text
        assert "Get-Command -Name $CondaExecutable" not in text


def test_qa_launcher_isolates_the_conda_environment() -> None:
    assert "$env:PYTHONNOUSERSITE = '1'" in _text(RUN_ALL_CHECKS)


def test_windows_docs_cover_conda_usage_boundaries_and_required_failures() -> None:
    readme = _text(README)
    guide = _text(USER_GUIDE)
    troubleshooting = _text(TROUBLESHOOTING)

    assert "Conda" in readme
    assert "not yet been completed" in readme
    guide_lower = guide.lower()
    for required in ("data preparation", "safe smoke", "full experiment queue", "desktop demonstrator", "result locations"):
        assert required in guide_lower
    assert ">= 0.80" in guide
    for required in ("No CUDA", "out of memory", "UNC", "weights", "video", "Invalid YAML"):
        assert required in troubleshooting
