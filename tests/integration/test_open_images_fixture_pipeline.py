"""End-to-end local-fixture checks for the Open Images CLI."""

from __future__ import annotations

import json
from pathlib import Path

from fruit_ssod.cli.download_open_images import main


FIXTURES = Path(__file__).parents[1] / "fixtures" / "open_images"


def test_cli_converts_and_downloads_checked_in_local_fixture_without_network(tmp_path: Path) -> None:
    """The CLI uses file: fixture URLs and never makes an HTTP request."""
    image_fixture = FIXTURES / "tiny.svg"
    output_root = tmp_path / "prepared"

    exit_code = main(
        [
            "--annotations", str(FIXTURES / "annotations.csv"),
            "--class-descriptions", str(FIXTURES / "class-descriptions.csv"),
            "--image-url-map", str(FIXTURES / "image-urls.csv"),
            "--output-root", str(output_root),
            "--max-images", "2",
            "--download",
        ]
    )

    assert exit_code == 0
    assert (output_root / "images" / "img-apple.svg").read_bytes() == image_fixture.read_bytes()
    assert (output_root / "labels" / "img-apple.txt").exists()
    manifest = [json.loads(line) for line in (output_root / "manifest.jsonl").read_text().splitlines()]
    assert [item["source_image_id"] for item in manifest] == ["img-apple", "img-orange"]


def test_cli_dry_run_creates_no_output_or_network_side_effects(tmp_path: Path) -> None:
    """Dry-run reports planned work but does not create its output directory."""
    output_root = tmp_path / "dry-output"

    exit_code = main(
        [
            "--annotations", str(FIXTURES / "annotations.csv"),
            "--class-descriptions", str(FIXTURES / "class-descriptions.csv"),
            "--image-url-map", str(FIXTURES / "image-urls.csv"),
            "--output-root", str(output_root),
            "--dry-run",
            "--max-images", "1",
        ]
    )

    assert exit_code == 0
    assert not output_root.exists()
