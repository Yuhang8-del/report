"""Offline-safe Open Images CSV conversion and explicit image acquisition.

Author: Fruit SSOD contributors
Date: 2026-07-31
Version: 1.0.0
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fruit_ssod.data.class_mapping import ClassMappingError, resolve_class_id
from fruit_ssod.data.yolo_format import YoloFormatError, format_yolo_label, xyxy_normalized_to_yolo


SOURCE_NAME = "open_images_v7"
"""Registry key for Open Images labels approved by the canonical class map."""


class OpenImagesConversionError(ValueError):
    """Raised when Open Images input cannot be safely converted."""


class DownloadError(RuntimeError):
    """Raised when an explicit image acquisition cannot complete safely."""


@dataclass(frozen=True)
class ConvertedImage:
    """A converted image identity, URL, and immutable set of generated labels."""

    source_image_id: str
    url: str
    labels: tuple[str, ...]
    class_ids: tuple[int, ...]


@dataclass(frozen=True)
class ConversionResult:
    """Summary returned by conversion, including intentionally excluded records."""

    images: tuple[ConvertedImage, ...]
    filtered_flagged_rows: int
    rejected_invalid_boxes: int


@dataclass(frozen=True)
class DownloadRecord:
    """An append-only ledger record for one source image."""

    source_image_id: str
    url: str
    file_path: Path
    sha256: str
    status: str


UrlOpener = Callable[[Request, float], object]


def _problem(problem: str, cause: str, remediation: str) -> str:
    """Build useful failures that can be acted on without inspecting source code."""
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _require_columns(fieldnames: list[str] | None, required: set[str], path: Path) -> None:
    """Reject ambiguous CSV schemas before rows are processed."""
    missing = sorted(required.difference(fieldnames or []))
    if missing:
        raise OpenImagesConversionError(
            _problem(
                f"CSV file {path} is missing required columns {missing}",
                "the input is not the expected Open Images export or manifest",
                "provide a UTF-8 CSV with the documented headers",
            )
        )


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Read a UTF-8 CSV entirely so malformed inputs fail before output writes."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_columns(reader.fieldnames, required, path)
            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                materialized = dict(row)
                for column in required:
                    value = materialized.get(column)
                    if not isinstance(value, str) or not value.strip():
                        raise OpenImagesConversionError(
                            _problem(
                                f"CSV file {path} row {row_number} has an empty or truncated {column!r} field",
                                "a required value is missing or the row has fewer fields than its header",
                                "repair the CSV row so every documented required column has text",
                            )
                        )
                rows.append(materialized)
            return rows
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise OpenImagesConversionError(
            _problem(
                f"CSV file {path} could not be read",
                str(error),
                "verify the path and use a readable UTF-8 comma-separated file",
            )
        ) from error


def read_image_url_map(path: Path) -> dict[str, str]:
    """Load a configurable image manifest, preserving first-seen deterministic order."""
    rows = _read_csv(path, {"ImageID", "OriginalURL"})
    urls: dict[str, str] = {}
    for row in rows:
        image_id = row["ImageID"].strip()
        url = row["OriginalURL"].strip()
        if not image_id or not url:
            raise OpenImagesConversionError(
                _problem(
                    f"image URL map {path} has an empty ImageID or OriginalURL",
                    "a manifest row is incomplete",
                    "supply a unique source image ID and URL in every selected row",
                )
            )
        if image_id in urls and urls[image_id] != url:
            raise OpenImagesConversionError(
                _problem(
                    f"image URL map has conflicting URLs for source image ID {image_id!r}",
                    "the manifest duplicated an image ID with different source URLs",
                    "deduplicate the image manifest before conversion",
                )
            )
        parsed = urlparse(url)
        if parsed.scheme == "file" and not parsed.netloc and parsed.path and not Path(parsed.path).is_absolute():
            url = (path.parent / parsed.path).resolve().as_uri()
        urls.setdefault(image_id, url)
    return urls


def _read_class_descriptions(path: Path) -> dict[str, str]:
    """Map Open Images machine label IDs to their exact reviewed display names."""
    # The official V7 ``oidv7-class-descriptions-boxable.csv`` deliberately
    # has no header (it is simply ``LabelName,DisplayName`` per line), while
    # the small local fixtures and some exported variants do have one.  Keep
    # both forms valid so the public official file can be used directly.
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            parsed = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise OpenImagesConversionError(
            _problem(
                f"class descriptions {path} could not be read",
                str(error),
                "download the official Open Images class-description CSV again",
            )
        ) from error
    if not parsed:
        raise OpenImagesConversionError(
            _problem(
                f"class descriptions {path} is empty",
                "there are no Open Images label mappings",
                "provide the official boxable class-description CSV",
            )
        )
    if parsed[0] == ["LabelName", "DisplayName"]:
        parsed = parsed[1:]
    rows = [{"LabelName": row[0], "DisplayName": row[1]} for row in parsed if len(row) >= 2]
    if not rows or any(len(row) < 2 for row in parsed):
        raise OpenImagesConversionError(
            _problem(
                f"class descriptions {path} has malformed rows",
                "each row must contain a label MID and display name",
                "use the unmodified official CSV or a two-column UTF-8 export",
            )
        )
    classes: dict[str, str] = {}
    for row in rows:
        label_id = row["LabelName"].strip()
        display_name = row["DisplayName"].strip()
        if not label_id or not display_name:
            raise OpenImagesConversionError(
                _problem(
                    f"class descriptions {path} has an empty LabelName or DisplayName",
                    "a category row is incomplete",
                    "supply exact Open Images label IDs and reviewed display names",
                )
            )
        if label_id in classes and classes[label_id] != display_name:
            raise OpenImagesConversionError(
                _problem(
                    f"class descriptions conflict for LabelName {label_id!r}",
                    "the CSV contains inconsistent duplicate category rows",
                    "deduplicate class descriptions before conversion",
                )
            )
        classes.setdefault(label_id, display_name)
    return classes


def _is_true(value: str) -> bool:
    """Interpret Open Images boolean exports without accepting arbitrary values."""
    return value.strip().lower() in {"1", "true", "t", "yes"}


def _safe_image_stem(source_image_id: str) -> str:
    """Keep source IDs exact while preventing a manifest from escaping output_root."""
    if not source_image_id or Path(source_image_id).name != source_image_id or source_image_id in {".", ".."}:
        raise DownloadError(
            _problem(
                f"source image ID {source_image_id!r} is unsafe for a local filename",
                "the manifest contains a path separator or traversal-like identifier",
                "use Open Images source IDs without path components",
            )
        )
    return source_image_id


def _url_suffix(url: str) -> str:
    """Use a conservative extension derived from the configured URL, defaulting to jpg."""
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".svg"} else ".jpg"


def _validate_download_url(url: str, source_image_id: str) -> None:
    """Reject malformed manifests before urllib raises an unhelpful ValueError."""
    parsed = urlparse(url)
    supported = {"http", "https", "file"}
    valid = parsed.scheme in supported and (
        (parsed.scheme in {"http", "https"} and bool(parsed.netloc))
        or (parsed.scheme == "file" and parsed.netloc in {"", "localhost"} and bool(parsed.path))
    )
    if not valid:
        raise DownloadError(
            _problem(
                f"configured URL for source image ID {source_image_id!r} is invalid",
                "the image manifest uses an empty, relative, unsupported, or remote-authority file URL",
                "use an absolute http(s) URL or a local file:// URL with empty authority (or localhost), then rerun",
            )
        )


def _content_range_start(response: object, source_image_id: str) -> int:
    """Read and validate the start byte from a range response before appending."""
    headers = getattr(response, "headers", None)
    content_range = headers.get("Content-Range") if hasattr(headers, "get") else None
    match = re.fullmatch(r"bytes\s+(\d+)-\d+/(?:\d+|\*)", content_range or "")
    if match is None:
        raise DownloadError(
            _problem(
                f"range response for source image ID {source_image_id!r} has malformed Content-Range",
                "the server returned 206 without a usable byte-range header",
                "retry against a Range-compliant source; the existing .part file was retained",
            )
        )
    return int(match.group(1))


def convert_open_images(
    annotations_path: Path,
    class_descriptions_path: Path,
    image_url_map_path: Path,
    output_root: Path,
    *,
    max_images: int | None = None,
    dry_run: bool = False,
) -> ConversionResult:
    """Convert approved Open Images rows to YOLO labels under ``output_root`` only."""
    if max_images is not None and max_images <= 0:
        raise OpenImagesConversionError(
            _problem("max_images must be positive", "a zero or negative limit was supplied", "omit it or provide a positive integer")
        )
    descriptions = _read_class_descriptions(class_descriptions_path)
    urls = read_image_url_map(image_url_map_path)
    rows = _read_csv(
        annotations_path,
        {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsDepiction", "IsInside", "IsGroupOf"},
    )
    images: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
    flagged = 0
    rejected = 0
    for row in rows:
        if any(_is_true(row[name]) for name in ("IsDepiction", "IsInside", "IsGroupOf")):
            flagged += 1
            continue
        image_id = row["ImageID"].strip()
        label_id = row["LabelName"].strip()
        if not image_id or not label_id:
            raise OpenImagesConversionError(
                _problem("annotation has an empty ImageID or LabelName", "a source row is incomplete", "repair the Open Images annotation CSV")
            )
        display_name = descriptions.get(label_id)
        if display_name is None:
            raise OpenImagesConversionError(
                _problem(
                    f"annotation label {label_id!r} is absent from class descriptions",
                    "the annotation and class CSV files do not come from the same export",
                    "use matching Open Images CSV inputs",
                )
            )
        try:
            class_id = resolve_class_id(SOURCE_NAME, display_name)
            box = xyxy_normalized_to_yolo(*(float(row[name]) for name in ("XMin", "YMin", "XMax", "YMax")))
        except (ValueError, ClassMappingError, YoloFormatError) as error:
            if isinstance(error, YoloFormatError):
                rejected += 1
                continue
            raise OpenImagesConversionError(
                _problem(
                    f"annotation for source image ID {image_id!r} cannot be converted",
                    str(error),
                    "use reviewed Open Images categories and finite normalized box coordinates",
                )
            ) from error
        if image_id not in urls:
            raise OpenImagesConversionError(
                _problem(
                    f"selected source image ID {image_id!r} has no URL manifest entry",
                    "the URL map is incomplete",
                    "add the image ID and its source URL to --image-url-map",
                )
            )
        if image_id not in images:
            if max_images is not None and len(images) >= max_images:
                continue
            images[image_id] = []
        images[image_id].append((class_id, format_yolo_label(class_id, box)))
    converted = tuple(
        ConvertedImage(
            source_image_id=image_id,
            url=urls[image_id],
            labels=tuple(label for _, label in labels),
            class_ids=tuple(class_id for class_id, _ in labels),
        )
        for image_id, labels in images.items()
    )
    result = ConversionResult(converted, flagged, rejected)
    if not dry_run:
        _write_conversion(result, output_root)
    return result


def _write_conversion(result: ConversionResult, output_root: Path) -> None:
    """Write labels and a source-ID manifest only beneath the explicit destination."""
    labels_dir = output_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[str] = []
    for image in result.images:
        stem = _safe_image_stem(image.source_image_id)
        (labels_dir / f"{stem}.txt").write_text("\n".join(image.labels) + "\n", encoding="utf-8")
        manifest_rows.append(json.dumps({"source": SOURCE_NAME, "source_image_id": image.source_image_id, "url": image.url, "class_ids": list(image.class_ids)}, sort_keys=True))
    (output_root / "manifest.jsonl").write_text("\n".join(manifest_rows) + ("\n" if manifest_rows else ""), encoding="utf-8")


def _default_opener(request: Request, timeout: float) -> object:
    """Keep urllib use isolated so tests can supply a non-network opener."""
    return urlopen(request, timeout=timeout)


def _sha256(path: Path) -> str:
    """Compute a content checksum for an image recorded in the download ledger."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_ledger(path: Path, record: DownloadRecord) -> None:
    """Append a durable, source-ID-preserving acquisition event."""
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps({"source": SOURCE_NAME, "source_image_id": record.source_image_id, "url": record.url, "file_path": str(record.file_path), "sha256": record.sha256, "status": record.status}, sort_keys=True) + "\n")


def download_images(
    image_urls: Mapping[str, str],
    output_root: Path,
    *,
    opener: UrlOpener = _default_opener,
    timeout: float = 30.0,
    workers: int = 1,
    retries: int = 2,
) -> tuple[DownloadRecord, ...]:
    """Explicitly download manifest images with Range resume and atomic promotion."""
    if workers <= 0:
        raise DownloadError(_problem("download workers must be positive", "a zero or negative worker count was supplied", "use one or more download workers"))
    if retries < 0:
        raise DownloadError(_problem("download retries must be nonnegative", "a negative retry count was supplied", "use zero for one attempt or a positive retry count"))
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "download-ledger.jsonl"
    items = tuple(image_urls.items())
    for source_image_id, url in items:
        _validate_download_url(url, source_image_id)

    def download_one(source_image_id: str, url: str) -> DownloadRecord:
        stem = _safe_image_stem(source_image_id)
        destination = image_dir / f"{stem}{_url_suffix(url)}"
        if destination.exists():
            return DownloadRecord(source_image_id, url, destination, _sha256(destination), "existing")
        partial = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(retries + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            try:
                request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            except ValueError as error:
                raise DownloadError(
                    _problem(
                        f"configured URL for source image ID {source_image_id!r} is invalid",
                        str(error),
                        "use an absolute http(s) URL or a valid file:// URL, then rerun",
                    )
                ) from error
            try:
                response = opener(request, timeout)
                with response as stream:  # type: ignore[union-attr]
                    status = getattr(stream, "status", None)
                    append = offset > 0 and status == 206
                    if append and _content_range_start(stream, source_image_id) != offset:
                        raise DownloadError(
                            _problem(
                                f"range response for source image ID {source_image_id!r} starts at the wrong offset",
                                f"the retained .part file has {offset} bytes but the server responded with a different start byte",
                                "retry against a Range-compliant source; the existing .part file was retained",
                            )
                        )
                    mode = "ab" if append else "wb"
                    with partial.open(mode) as handle:
                        while True:
                            block = stream.read(1024 * 1024)
                            if not block:
                                break
                            handle.write(block)
                break
            except DownloadError:
                raise
            except (OSError, HTTPError, URLError, ValueError) as error:
                if attempt == retries:
                    raise DownloadError(
                        _problem(
                            f"download failed for source image ID {source_image_id!r}",
                            str(error),
                            "check the configured URL and connectivity, then rerun; the .part file is retained for resume",
                        )
                    ) from error
        try:
            os.replace(partial, destination)
            return DownloadRecord(source_image_id, url, destination, _sha256(destination), "downloaded")
        except OSError as error:
            raise DownloadError(
                _problem(
                    f"download could not be promoted for source image ID {source_image_id!r}",
                    str(error),
                    "verify write permission and free space, then rerun to resume the retained .part file",
                )
            ) from error
    if workers == 1 or len(items) <= 1:
        records = [download_one(source_image_id, url) for source_image_id, url in items]
    else:
        records_by_id: dict[str, DownloadRecord] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="open-images-download") as executor:
            futures = {executor.submit(download_one, source_image_id, url): source_image_id for source_image_id, url in items}
            for future in as_completed(futures):
                records_by_id[futures[future]] = future.result()
        records = [records_by_id[source_image_id] for source_image_id, _ in items]
    for record in records:
        _append_ledger(ledger_path, record)
    return tuple(records)
