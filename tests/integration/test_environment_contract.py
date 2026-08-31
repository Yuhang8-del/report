"""Integration contract for the Windows environment preflight script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = REPOSITORY_ROOT / "scripts" / "preflight.ps1"
ANACONDA_PYTHON = Path(r"E:\anaconda\python.exe")


def run_preflight(
    data_root: Path | str,
    artifact_root: Path,
    *,
    skip_data_root_reachability: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the native PowerShell preflight against caller-controlled local paths."""
    powershell = os.environ.get("POWERSHELL_EXE", "powershell.exe")
    command = [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PREFLIGHT),
            "-PythonExecutable",
            str(ANACONDA_PYTHON),
            "-DataRoot",
            str(data_root),
            "-ArtifactRoot",
            str(artifact_root),
            "-ReachabilityTimeoutSeconds",
            "2",
        ]
    if skip_data_root_reachability:
        command.append("-SkipDataRootReachability")
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def test_preflight_succeeds_with_explicit_local_roots(tmp_path: Path) -> None:
    """Local roots make this test independent from the approved UNC share."""
    data_root = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    data_root.mkdir()
    artifact_root.mkdir()

    result = run_preflight(data_root, artifact_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS] Python executable" in result.stdout
    assert "[PASS] Shared data root reachability" in result.stdout
    assert "[PASS] Artifact write/delete probe" in result.stdout
    assert "Overall result: PASS" in result.stdout
    assert not list(artifact_root.iterdir())


def test_preflight_fails_for_missing_data_root(tmp_path: Path) -> None:
    """A missing configured root must fail instead of falling back silently."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    result = run_preflight(tmp_path / "missing-data", artifact_root)

    assert result.returncode != 0
    assert "[FAIL] Shared data root reachability" in result.stdout
    assert "Likely cause:" in result.stdout
    assert "Remediation:" in result.stdout
    assert "Overall result: FAIL" in result.stdout


def test_preflight_skip_does_not_hide_a_missing_local_data_root(tmp_path: Path) -> None:
    """The offline switch is exclusively for unavailable UNC shares."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    result = run_preflight(
        tmp_path / "missing-local-data",
        artifact_root,
        skip_data_root_reachability=True,
    )

    assert result.returncode != 0
    assert "[FAIL] Shared data root reachability" in result.stdout
    assert "[SKIP] Shared data root reachability" not in result.stdout
    assert "[SKIP] Configured data root free space" not in result.stdout


def test_preflight_skip_treats_extended_local_paths_as_local(tmp_path: Path) -> None:
    """``\\\\?\\C:\\...`` is a local path, unlike extended UNC syntax."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    extended_missing = "\\\\?\\" + str(tmp_path / "missing-extended-local-data")

    result = run_preflight(
        extended_missing,
        artifact_root,
        skip_data_root_reachability=True,
    )

    assert result.returncode != 0
    assert "[PASS] Configured data root free space" in result.stdout
    assert "[FAIL] Shared data root reachability" in result.stdout
    assert "[SKIP] Shared data root reachability" not in result.stdout


def test_preflight_skip_bypasses_unc_data_root_checks_but_not_artifact_checks(
    tmp_path: Path,
) -> None:
    """Offline diagnostics must not contact the configured UNC data root."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    result = run_preflight(
        r"\\10.16.57.94\dataset2\lyg\detect_datasets",
        artifact_root,
        skip_data_root_reachability=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[SKIP] Shared data root reachability" in result.stdout
    assert "[SKIP] Configured data root free space" in result.stdout
    assert "[PASS] Configured artifact root free space" in result.stdout
    assert "[PASS] Artifact write/delete probe" in result.stdout
    assert "Overall result: PASS" in result.stdout
    assert not list(artifact_root.iterdir())
