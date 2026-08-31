"""Unit tests for safe, source-aware Open Images conversion."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from fruit_ssod.data.open_images import (
    DownloadError,
    OpenImagesConversionError,
    convert_open_images,
    download_images,
    read_image_url_map,
)
from fruit_ssod.data.open_images_selection import build_open_images_selection
from fruit_ssod.data.yolo_format import YoloFormatError, xyxy_normalized_to_yolo


FIXTURES = Path(__file__).parents[1] / "fixtures" / "open_images"


def test_conversion_uses_registry_filters_flags_and_clamps_boxes(tmp_path: Path) -> None:
    """Only reviewed, non-flagged objects become stable YOLO labels."""
    result = convert_open_images(
        annotations_path=FIXTURES / "annotations.csv",
        class_descriptions_path=FIXTURES / "class-descriptions.csv",
        image_url_map_path=FIXTURES / "image-urls.csv",
        output_root=tmp_path / "converted",
    )

    assert [item.source_image_id for item in result.images] == ["img-apple", "img-orange"]
    assert result.filtered_flagged_rows == 3
    assert result.rejected_invalid_boxes == 1
    assert (tmp_path / "converted" / "labels" / "img-apple.txt").read_text() == "0 0.300000 0.400000 0.400000 0.400000\n"
    assert (tmp_path / "converted" / "labels" / "img-orange.txt").read_text() == "2 0.500000 0.500000 1.000000 0.500000\n"
    manifest = [json.loads(line) for line in (tmp_path / "converted" / "manifest.jsonl").read_text().splitlines()]
    assert [row["source_image_id"] for row in manifest] == ["img-apple", "img-orange"]
    assert [row["class_ids"] for row in manifest] == [[0], [2]]


def test_max_images_is_deterministic_and_dry_run_does_not_write(tmp_path: Path) -> None:
    """Dry-run has no output side effects and limits unique source IDs by CSV order."""
    output_root = tmp_path / "output"
    dry_result = convert_open_images(
        annotations_path=FIXTURES / "annotations.csv",
        class_descriptions_path=FIXTURES / "class-descriptions.csv",
        image_url_map_path=FIXTURES / "image-urls.csv",
        output_root=output_root,
        max_images=1,
        dry_run=True,
    )

    assert [item.source_image_id for item in dry_result.images] == ["img-apple"]
    assert not output_root.exists()


def test_unknown_open_images_label_fails_with_actionable_error(tmp_path: Path) -> None:
    """An unreviewed Open Images display name can never receive a guessed class ID."""
    classes = tmp_path / "classes.csv"
    classes.write_text("LabelName,DisplayName\n/m/apple,Apple\n/m/unknown,Unknown Fruit\n", encoding="utf-8")
    annotations = tmp_path / "annotations.csv"
    annotations.write_text(
        "ImageID,LabelName,XMin,XMax,YMin,YMax,IsDepiction,IsInside,IsGroupOf\n"
        "image-1,/m/unknown,0.1,0.5,0.2,0.6,0,0,0\n",
        encoding="utf-8",
    )
    urls = tmp_path / "urls.csv"
    urls.write_text("ImageID,OriginalURL\nimage-1,https://fixture.invalid/image-1.jpg\n", encoding="utf-8")

    with pytest.raises(OpenImagesConversionError, match="Problem:") as error:
        convert_open_images(annotations, classes, urls, tmp_path / "out")

    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_yolo_conversion_rejects_zero_area_after_clamping() -> None:
    """A box outside an edge must not turn into a zero-area YOLO annotation."""
    with pytest.raises(YoloFormatError, match="non-zero area"):
        xyxy_normalized_to_yolo(1.1, 0.2, 1.2, 0.6)


class _Response(io.BytesIO):
    """Small context-managed URL response used without an HTTP server."""

    def __init__(self, payload: bytes, status: int, headers: dict[str, str]) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_download_resumes_part_file_records_sha_and_never_overwrites_source(tmp_path: Path) -> None:
    """A range-capable response finishes a partial image through atomic promotion."""
    payload = b"complete-local-image-bytes"
    image_id = "img-apple"
    urls = tmp_path / "urls.csv"
    urls.write_text(f"ImageID,OriginalURL\n{image_id},https://fixture.invalid/{image_id}.jpg\n", encoding="utf-8")
    output_root = tmp_path / "output"
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True)
    (image_dir / f"{image_id}.jpg.part").write_bytes(payload[:8])
    requests: list[object] = []

    def opener(request: object, timeout: float) -> _Response:
        requests.append(request)
        return _Response(payload[8:], 206, {"Content-Range": f"bytes 8-{len(payload) - 1}/{len(payload)}"})

    records = download_images(read_image_url_map(urls), output_root, opener=opener)

    image_path = image_dir / f"{image_id}.jpg"
    assert image_path.read_bytes() == payload
    assert not image_path.with_suffix(".jpg.part").exists()
    assert requests[0].get_header("Range") == "bytes=8-"
    assert records[0].sha256 == hashlib.sha256(payload).hexdigest()
    ledger = [json.loads(line) for line in (output_root / "download-ledger.jsonl").read_text().splitlines()]
    assert ledger[0]["source_image_id"] == image_id

    def failing_opener(request: object, timeout: float) -> _Response:
        raise AssertionError("existing source image must not be fetched or overwritten")

    repeated = download_images(read_image_url_map(urls), output_root, opener=failing_opener)
    assert repeated[0].status == "existing"
    assert image_path.read_bytes() == payload


def test_download_errors_are_actionable(tmp_path: Path) -> None:
    """Transport failures tell the operator how to retry safely."""
    with pytest.raises(DownloadError, match="Problem:") as error:
        download_images(
            {"img-1": "https://fixture.invalid/img-1.jpg"},
            tmp_path / "output",
            opener=lambda request, timeout: (_ for _ in ()).throw(URLError("offline")),
        )
    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_download_retries_transient_transport_failure(tmp_path: Path) -> None:
    """One transient transport failure is retried before the image is abandoned."""
    attempts = 0

    def opener(request: object, timeout: float) -> _Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("temporary TLS disconnect")
        return _Response(b"recovered-image", 200, {})

    records = download_images(
        {"img-1": "https://fixture.invalid/img-1.jpg"},
        tmp_path / "output",
        opener=opener,
        retries=1,
    )

    assert attempts == 2
    assert records[0].status == "downloaded"
    assert (tmp_path / "output" / "images" / "img-1.jpg").read_bytes() == b"recovered-image"


def test_parallel_download_keeps_manifest_order_and_one_ledger_row_per_image(tmp_path: Path) -> None:
    """Workers may finish in any order, but published evidence remains deterministic."""
    payloads = {"a": b"a-bytes", "b": b"b-bytes"}

    def opener(request: object, timeout: float) -> _Response:
        image_id = str(getattr(request, "full_url")).rsplit("/", 1)[-1].split(".")[0]
        return _Response(payloads[image_id], 200, {})

    records = download_images({"a": "https://fixture.invalid/a.jpg", "b": "https://fixture.invalid/b.jpg"}, tmp_path / "parallel", opener=opener, workers=2)

    assert [record.source_image_id for record in records] == ["a", "b"]
    ledger = [json.loads(line)["source_image_id"] for line in (tmp_path / "parallel" / "download-ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ledger == ["a", "b"]


def test_download_rejects_mismatched_content_range_and_retains_partial_file(tmp_path: Path) -> None:
    """A 206 response must begin exactly after the retained local bytes."""
    payload = b"complete-local-image-bytes"
    output_root = tmp_path / "output"
    partial = output_root / "images" / "img-apple.jpg.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:8])

    with pytest.raises(DownloadError, match="Problem:") as error:
        download_images(
            {"img-apple": "https://fixture.invalid/img-apple.jpg"},
            output_root,
            opener=lambda request, timeout: _Response(
                payload[8:], 206, {"Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}"}
            ),
        )

    assert "Likely cause:" in str(error.value)
    assert partial.read_bytes() == payload[:8]
    assert not (output_root / "images" / "img-apple.jpg").exists()


def test_malformed_url_is_rejected_as_an_actionable_download_error(tmp_path: Path) -> None:
    """Bad manifest URLs fail before urllib can leak a raw ValueError."""
    with pytest.raises(DownloadError, match="Problem:") as error:
        download_images({"img-1": "not-a-url"}, tmp_path / "output")

    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_file_url_with_remote_authority_is_rejected(tmp_path: Path) -> None:
    """A file URL must not redirect local-fixture logic onto a host or share."""
    with pytest.raises(DownloadError, match="Problem:") as error:
        download_images({"img-1": "file://untrusted-host/share/image.jpg"}, tmp_path / "output")

    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


@pytest.mark.parametrize("truncated_kind", ["class_descriptions", "url_map", "annotations"])
def test_truncated_csv_rows_raise_actionable_conversion_errors(tmp_path: Path, truncated_kind: str) -> None:
    """DictReader's None cells are reported as bad source rows, never AttributeError."""
    annotations = FIXTURES / "annotations.csv"
    descriptions = FIXTURES / "class-descriptions.csv"
    urls = FIXTURES / "image-urls.csv"
    if truncated_kind == "class_descriptions":
        descriptions = tmp_path / "classes.csv"
        descriptions.write_text("LabelName,DisplayName\n/m/apple\n", encoding="utf-8")
    elif truncated_kind == "url_map":
        urls = tmp_path / "urls.csv"
        urls.write_text("ImageID,OriginalURL\nimg-apple\n", encoding="utf-8")
    else:
        annotations = tmp_path / "annotations.csv"
        annotations.write_text(
            "ImageID,LabelName,XMin,XMax,YMin,YMax,IsDepiction,IsInside,IsGroupOf\n"
            "img-apple,/m/apple,0.1,0.5,0.2,0.6,0,0\n",
            encoding="utf-8",
        )

    with pytest.raises(OpenImagesConversionError, match="Problem:") as error:
        convert_open_images(annotations, descriptions, urls, tmp_path / "out")

    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_selection_accepts_official_headerless_class_descriptions_and_preserves_url_provenance(tmp_path: Path) -> None:
    """The public V7 class CSV is headerless, unlike the small checked-in fixture."""
    classes = tmp_path / "classes.csv"
    classes.write_text("/a,Apple\n/b,Banana\n/c,Orange\n/d,Strawberry\n/e,Pineapple\n", encoding="utf-8")
    annotations = tmp_path / "annotations.csv"
    annotations.write_text(
        "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"
        "one,xclick,/a,1,0.1,0.8,0.1,0.8,0,0,0,0,0\n"
        "two,xclick,/b,1,0.1,0.8,0.1,0.8,0,0,0,0,0\n"
        "three,xclick,/c,1,0.1,0.8,0.1,0.8,0,0,0,0,0\n"
        "four,xclick,/d,1,0.1,0.8,0.1,0.8,0,0,0,0,0\n"
        "five,xclick,/e,1,0.1,0.8,0.1,0.8,0,0,0,0,0\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "ImageID,OriginalURL,License,Author,AuthorProfileURL,OriginalLandingURL,Subset,Thumbnail300KURL\n"
        + "\n".join(f"{image_id},https://example.invalid/{image_id}.jpg,CC-BY,Author,https://example.invalid/a,https://example.invalid/p,train,https://thumb.example.invalid/{image_id}.jpg" for image_id in ("one", "two", "three", "four", "five"))
        + "\n",
        encoding="utf-8",
    )

    result = build_open_images_selection(annotations, classes, metadata, tmp_path / "selection", per_class=1)

    assert result.class_image_counts == {"Apple": 1, "Banana": 1, "Orange": 1, "Strawberry": 1, "Pineapple": 1}
    output = (tmp_path / "selection" / "image-urls.csv").read_text(encoding="utf-8")
    assert len(output.splitlines()) == 6
    assert "https://open-images-dataset.s3.amazonaws.com/train/one.jpg" in output
    assert "https://thumb.example.invalid/one.jpg" in output
    assert "https://example.invalid/one.jpg" in output
