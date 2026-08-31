"""Atomic, local-only export packages for file-inference results."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from fruit_ssod.gui.workers.image_worker import ImageInferenceResult


class ResultExportError(RuntimeError):
    """An actionable error raised when an inference export cannot be produced."""


def _error(problem: str, cause: str, remediation: str) -> ResultExportError:
    return ResultExportError(
        f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."
    )


_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (220, 45, 45),
    1: (240, 190, 25),
    2: (245, 130, 35),
    3: (210, 55, 100),
    4: (120, 155, 35),
}


@dataclass(frozen=True)
class ExportManifest:
    """Exact local paths atomically published by one export request."""

    output_dir: Path
    annotated_images: tuple[Path, ...]
    csv_path: Path
    json_path: Path
    manifest_path: Path


def _atomic_write_text(destination: Path, text: str) -> None:
    """Write a staged metadata file without exposing a torn file in the package."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _safe_image_name(result: ImageInferenceResult, index: int) -> str:
    """Avoid filename collisions when a batch contains duplicate basenames."""
    stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in result.image_path.stem
    )
    return f"{index:04d}_{stem or 'image'}_annotated.png"


def _annotated_image(result: ImageInferenceResult, destination: Path) -> None:
    """Render known-fruit boxes without using model-specific result objects."""
    try:
        with Image.open(result.image_path) as source:
            image = source.convert("RGB")
    except Exception as error:
        raise _error(
            "annotated image could not be created",
            f"{result.image_path.name} could not be decoded: {error}",
            "verify that the selected file is a readable image and rerun inference",
        ) from error
    drawing = ImageDraw.Draw(image)
    for detection in result.detections:
        color = _CLASS_COLORS[detection.class_id]
        drawing.rectangle(detection.xyxy, outline=color, width=3)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        left, top, _right, _bottom = detection.xyxy
        drawing.text((left + 2, max(0.0, top - 14)), label, fill=color)
    temporary = destination.with_suffix(".tmp.png")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _write_csv(rows: list[dict[str, object]], destination: Path) -> None:
    columns = (
        "image_path", "annotated_image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
        "latency_ms", "confidence_threshold", "nms_iou_threshold",
    )
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(destination, buffer.getvalue())


def _package_manifest(package_dir: Path, annotated_paths: list[Path]) -> dict[str, object]:
    """Capture the complete staged package before its single publish operation."""
    artifact_paths = [*annotated_paths, package_dir / "detections.csv", package_dir / "results.json"]
    artifacts: list[dict[str, object]] = []
    for path in artifact_paths:
        digest, size = _sha256_and_size(path)
        artifacts.append(
            {
                "path": str(path.relative_to(package_dir)).replace("\\", "/"),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return {
        "artifact_type": "fruit_ssod_image_inference_export_manifest",
        "version": 1,
        "artifacts": artifacts,
    }


def _publish_package(staging: Path, destination: Path) -> None:
    """Atomically name a complete package without replacing another export.

    ``os.rename`` is deliberately used instead of ``os.replace``: another
    process winning the race to create the requested destination must cause this
    export to fail, never silently replace or mix with an existing package.
    """
    if destination.exists():
        raise _error(
            "export destination already exists",
            f"{destination} already contains an export package",
            "choose a new empty export folder; existing packages are never overwritten",
        )
    try:
        os.rename(staging, destination)
    except OSError as error:
        if destination.exists():
            raise _error(
                "export destination was claimed by another export",
                f"{destination} appeared while this package was being published",
                "choose a different export folder and retry",
            ) from error
        raise _error(
            "completed export package could not be published",
            str(error),
            "choose a writable local parent folder and retry",
        ) from error


@contextmanager
def _publication_lock(destination: Path) -> Iterable[None]:
    """Serialize competing exports targeting one package directory.

    Directory rename is atomic, while the exclusive sibling lock closes the
    check-to-publish gap between two instances of this exporter.  The lock is
    deliberately outside the package so the published artifact set remains
    exactly the requested deliverable.
    """
    lock_path = destination.parent / f".{destination.name}.publish.lock"
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except FileExistsError as error:
        raise _error(
            "export destination is being published by another export",
            f"{lock_path.name} already exists",
            "wait for that export to finish or choose a different export folder",
        ) from error
    except OSError as error:
        raise _error(
            "export publication lock could not be created",
            str(error),
            "choose a writable local parent folder and retry",
        ) from error
    try:
        yield
    finally:
        try:
            os.close(descriptor)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def export_inference_results(
    results: Iterable[ImageInferenceResult], output_dir: str | Path
) -> ExportManifest:
    """Publish one all-or-nothing local result package.

    Annotated images, CSV, JSON, and a checksummed manifest are written to a
    uniquely named sibling staging directory.  The staging directory is renamed
    only after every artifact exists; an existing requested destination is never
    overwritten, even if a competing export creates it concurrently.
    """
    frozen_results = tuple(results)
    if not frozen_results:
        raise _error(
            "there are no completed inference results to export",
            "the selected image run produced no successful predictions",
            "run at least one image successfully before choosing an export folder",
        )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        cause = "an existing file" if not destination.is_dir() else "an existing export package"
        raise _error(
            "export destination already exists",
            f"{destination} is {cause}",
            "choose a new export folder; existing paths are never overwritten",
        )

    staging: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
        )
        staged_annotated_dir = staging / "annotated_images"
        staged_annotated_dir.mkdir()
        final_annotated_dir = destination / "annotated_images"
        annotated_paths: list[Path] = []
        rows: list[dict[str, object]] = []
        for index, result in enumerate(frozen_results, start=1):
            image_name = _safe_image_name(result, index)
            staged_annotated_path = staged_annotated_dir / image_name
            final_annotated_path = final_annotated_dir / image_name
            _annotated_image(result, staged_annotated_path)
            annotated_paths.append(final_annotated_path)
            if not result.detections:
                rows.append(
                    {
                        "image_path": str(result.image_path), "annotated_image": str(final_annotated_path),
                        "class_id": "", "class_name": "", "confidence": "", "x1": "", "y1": "", "x2": "", "y2": "",
                        "latency_ms": result.latency_ms, "confidence_threshold": result.confidence, "nms_iou_threshold": result.nms_iou,
                    }
                )
            for detection in result.detections:
                x1, y1, x2, y2 = detection.xyxy
                rows.append(
                    {
                        "image_path": str(result.image_path), "annotated_image": str(final_annotated_path),
                        "class_id": detection.class_id, "class_name": detection.class_name, "confidence": detection.confidence,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "latency_ms": result.latency_ms, "confidence_threshold": result.confidence, "nms_iou_threshold": result.nms_iou,
                    }
                )
        _write_csv(rows, staging / "detections.csv")
        payload = {
            "artifact_type": "fruit_ssod_image_inference_export",
            "version": 1,
            "annotated_images": [str(path) for path in annotated_paths],
            "results": [result.to_dict() for result in frozen_results],
        }
        _atomic_write_text(
            staging / "results.json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        staged_manifest = _package_manifest(
            staging,
            [staged_annotated_dir / path.name for path in annotated_paths],
        )
        _atomic_write_text(
            staging / "manifest.json",
            json.dumps(staged_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        with _publication_lock(destination):
            _publish_package(staging, destination)
        staging = None
    except ResultExportError:
        raise
    except OSError as error:
        raise _error(
            "export package could not be created",
            str(error),
            "choose a writable local parent folder and retry",
        ) from error
    except Exception as error:
        raise _error(
            "export package could not be completed",
            str(error),
            "verify the source images and retry with a new local export folder",
        ) from error
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    return ExportManifest(
        output_dir=destination,
        annotated_images=tuple(annotated_paths),
        csv_path=destination / "detections.csv",
        json_path=destination / "results.json",
        manifest_path=destination / "manifest.json",
    )
