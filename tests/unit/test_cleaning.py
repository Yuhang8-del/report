"""Tests for deterministic, non-destructive annotation cleaning."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from PIL import Image
import pytest

from fruit_ssod.cli.clean_dataset import main
import fruit_ssod.data.cleaning as cleaning
from fruit_ssod.data.cleaning import clean_manifest_rows, write_quarantine_manifest


FIXTURES = Path(__file__).parents[1] / "fixtures" / "cleaning"


def _row(file_path: str, xyxy: list[float], source_image_id: str = "image-1") -> dict[str, object]:
    return {
        "source": "open_images_v7",
        "source_category": "Apple",
        "source_image_id": source_image_id,
        "file_path": file_path,
        "width": 10,
        "height": 8,
        "class_id": 0,
        "xyxy": xyxy,
        "split": "train_pool",
        "label_status": "labeled",
        "license_metadata": {"name": "fixture"},
    }


def test_cleaning_decodes_images_clamps_boxes_and_quarantines_invalid_rows(tmp_path: Path) -> None:
    """Valid boxes are clamped; corrupt, zero-area, and non-finite rows are retained as records."""
    image = tmp_path / "valid.png"
    Image.new("RGB", (10, 8), "red").save(image)
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes((FIXTURES / "corrupt.jpg").read_bytes())
    rows = [
        _row("valid.png", [-2.0, 1.0, 14.0, 9.0], "clamped"),
        _row("valid.png", [2.0, 3.0, 2.0, 7.0], "zero-area"),
        _row("corrupt.jpg", [1.0, 1.0, 3.0, 4.0], "corrupt"),
        _row("missing.png", [1.0, 1.0, 3.0, 4.0], "missing"),
        _row("valid.png", [1.0, float("nan"), 3.0, 4.0], "non-finite"),
    ]

    result = clean_manifest_rows(rows, image_root=tmp_path)

    assert len(result.accepted) == 1
    assert result.accepted[0].xyxy == (0.0, 1.0, 10.0, 8.0)
    assert result.accepted[0].source == "open_images_v7"
    assert result.accepted[0].class_id == 0
    assert [record.reason_code for record in result.rejected] == [
        "BOX_NON_POSITIVE_AREA",
        "IMAGE_UNDECODABLE",
        "IMAGE_MISSING",
        "BOX_NON_FINITE",
    ]
    assert all("remediation" in record.details for record in result.rejected)

    quarantine = tmp_path / "out" / "quarantine.jsonl"
    write_quarantine_manifest(quarantine, result.rejected)
    written = [json.loads(line) for line in quarantine.read_text(encoding="utf-8").splitlines()]
    assert written[1]["source_image_id"] == "corrupt"
    assert corrupt.exists(), "cleaning must never remove or move a source file"


def test_cleaning_dry_run_is_in_memory_and_preserves_source_row_identity(tmp_path: Path) -> None:
    """Cleaning output only materializes when a caller explicitly asks a writer to do so."""
    image = tmp_path / "valid.png"
    Image.new("RGB", (6, 6), "green").save(image)
    source = _row("valid.png", [1.0, 1.0, 5.0, 5.0])

    result = clean_manifest_rows([source], image_root=tmp_path)

    assert result.accepted[0].source_image_id == source["source_image_id"]
    assert result.accepted[0].source_category == source["source_category"]
    assert not (tmp_path / "unrequested-output").exists()


def test_clean_dataset_cli_writes_only_explicit_outputs_and_dry_run_writes_nothing(tmp_path: Path) -> None:
    """The CLI reports valid quarantine work without treating it as an invocation failure."""
    image = tmp_path / "valid.png"
    Image.new("RGB", (10, 8), "red").save(image)
    input_manifest = tmp_path / "input.json"
    input_manifest.write_text(json.dumps([_row("valid.png", [1.0, 1.0, 4.0, 4.0])]), encoding="utf-8")
    output = tmp_path / "output" / "cleaned.json"
    quarantine = tmp_path / "output" / "quarantine.jsonl"

    assert main(["--input-manifest", str(input_manifest), "--output-manifest", str(output), "--quarantine-manifest", str(quarantine), "--image-root", str(tmp_path), "--dry-run"]) == 0
    assert not output.exists()
    assert not quarantine.exists()
    assert main(["--input-manifest", str(input_manifest), "--output-manifest", str(output), "--quarantine-manifest", str(quarantine), "--image-root", str(tmp_path)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["accepted_count"] == 1
    assert quarantine.exists()


def test_clean_dataset_cli_rejects_invalid_invocation_with_nonzero_exit(tmp_path: Path) -> None:
    """A negative perceptual threshold is an invocation error, not a quarantine finding."""
    with pytest.raises(SystemExit) as exit_status:
        main([
            "--input-manifest", str(tmp_path / "input.json"),
            "--output-manifest", str(tmp_path / "output.json"),
            "--quarantine-manifest", str(tmp_path / "quarantine.jsonl"),
            "--near-hash-threshold", "-1",
        ])
    assert exit_status.value.code == 2


def test_cleaning_quarantines_a_pillow_decompression_bomb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pillow safety limits result in a structured image rejection, not a raw exception."""
    image = tmp_path / "safe-placeholder.png"
    image.write_bytes(b"placeholder")

    def bomb(_: Path) -> object:
        raise Image.DecompressionBombError("synthetic bomb")

    monkeypatch.setattr(cleaning.Image, "open", bomb)
    result = clean_manifest_rows([_row("safe-placeholder.png", [1.0, 1.0, 4.0, 4.0], "bomb")], image_root=tmp_path)

    assert result.rejected[0].reason_code == "IMAGE_UNDECODABLE"
    assert "synthetic bomb" in result.rejected[0].details["likely_cause"]


def test_cleaning_quarantines_a_pillow_decompression_bomb_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pillow's warning-only safety signal is promoted to an actionable rejection."""
    image = tmp_path / "safe-placeholder.png"
    image.write_bytes(b"placeholder")

    def warning(_: Path) -> object:
        warnings.warn("synthetic bomb warning", Image.DecompressionBombWarning)
        raise AssertionError("warning should have been promoted to an exception")

    monkeypatch.setattr(cleaning.Image, "open", warning)
    result = clean_manifest_rows([_row("safe-placeholder.png", [1.0, 1.0, 4.0, 4.0], "bomb-warning")], image_root=tmp_path)

    assert result.rejected[0].reason_code == "IMAGE_UNDECODABLE"
    assert "synthetic bomb warning" in result.rejected[0].details["likely_cause"]


@pytest.mark.parametrize("collision", ["output_is_input", "quarantine_is_input", "outputs_match", "output_is_source", "quarantine_is_source"])
def test_clean_dataset_cli_rejects_output_collisions_before_writing(tmp_path: Path, collision: str) -> None:
    """Output destinations cannot overwrite the input manifest, one another, or source images."""
    source = tmp_path / "source.png"
    Image.new("RGB", (10, 8), "red").save(source)
    source_bytes = source.read_bytes()
    input_manifest = tmp_path / "input.json"
    original_manifest = json.dumps([_row("source.png", [1.0, 1.0, 4.0, 4.0])])
    input_manifest.write_text(original_manifest, encoding="utf-8")
    output = tmp_path / "output" / "cleaned.json"
    quarantine = tmp_path / "output" / "quarantine.jsonl"
    if collision == "output_is_input":
        output = input_manifest
    elif collision == "quarantine_is_input":
        quarantine = input_manifest
    elif collision == "outputs_match":
        quarantine = output
    elif collision == "output_is_source":
        output = source
    else:
        quarantine = source

    with pytest.raises(SystemExit) as exit_status:
        main(["--input-manifest", str(input_manifest), "--output-manifest", str(output), "--quarantine-manifest", str(quarantine), "--image-root", str(tmp_path)])

    assert exit_status.value.code == 2
    assert input_manifest.read_text(encoding="utf-8") == original_manifest
    assert source.read_bytes() == source_bytes
    assert not (tmp_path / "output").exists()


def test_packaging_declares_image_runtime_dependencies() -> None:
    """Installing the package exposes the direct Pillow/ImageHash imports used at runtime."""
    project = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"Pillow>=10.4,<12"' in project
    assert '"ImageHash>=4.3,<5"' in project
