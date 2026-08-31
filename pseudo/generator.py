"""Offline original/flip teacher inference with sealed unlabeled inputs."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from fruit_ssod.data.schema import AnnotationValidationError, UnlabeledImageRecord
from fruit_ssod.detection.adapter import DetectorAdapter, DetectorAdapterError
from fruit_ssod.detection.types import DetectionRecord
from fruit_ssod.pseudo.candidates import PseudoCandidate
from fruit_ssod.pseudo.transforms import TransformError, horizontal_flip_image, horizontal_flip_xyxy


class PseudoGenerationError(RuntimeError):
    """Raised when pseudo candidates cannot be safely generated or published."""


def _problem(problem: str, cause: str, remediation: str) -> PseudoGenerationError:
    return PseudoGenerationError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


_UNLABELED_MANIFEST_FIELDS = frozenset({"source", "source_image_id", "file_path", "width", "height", "split", "label_status", "license_metadata"})
_HIDDEN_LABEL_FIELDS = frozenset({
    "labels", "label", "annotations", "annotation", "annotation_path", "human_label",
    "human_labels", "class_id", "class_name", "xyxy", "source_category", "ground_truth",
})


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _problem(f"{field} must be a nonempty string", "pseudo-label provenance omitted a required identifier", f"provide a nonempty {field}")
    return value


def _safe_manifest_row(value: object) -> UnlabeledImageRecord:
    """Accept only an explicit no-label manifest record, never a generic mapping."""
    if not isinstance(value, Mapping):
        raise _problem("unlabeled input record is not an object", "the generator was not given an explicit unlabeled image manifest row", "supply JSON objects from the Task 8 unlabeled.json records array")
    keys = set(value)
    hidden = sorted(str(key) for key in keys & _HIDDEN_LABEL_FIELDS)
    unknown = sorted(str(key) for key in keys - _UNLABELED_MANIFEST_FIELDS)
    if hidden or unknown:
        names = hidden or unknown
        raise _problem("unlabeled input contains label-bearing or unsupported fields", f"the manifest includes {names!r}, which could expose hidden human labels", "supply only source, source_image_id, file_path, width, height, split, label_status, and license_metadata")
    try:
        return UnlabeledImageRecord.from_mapping(value)
    except (AnnotationValidationError, TypeError, ValueError) as error:
        raise _problem("unlabeled input record is invalid", str(error), "regenerate the Task 8 unlabeled.json manifest and do not attach annotation files") from error


def _id_set(value: object, *, field: str) -> frozenset[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _problem(f"{field} must be an array of image IDs", "the Task 8 split manifest is malformed", "regenerate split_manifest.json with create_splits")
    result = frozenset(_text(item, f"{field} item") for item in value)
    if len(result) != len(value):
        raise _problem(f"{field} contains duplicate image IDs", "the Task 8 split manifest does not uniquely seal image membership", "regenerate split_manifest.json with create_splits")
    return result


_CANONICAL_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_unlabeled_fingerprint(records: Sequence[UnlabeledImageRecord]) -> str:
    """Reproduce Task 8's exact fingerprint of its no-label membership payload.

    Task 8 intentionally excludes license metadata from this digest: it seals
    image identity, source/path, decoded dimensions, and the workflow state.
    Keep the record order untouched, because it is part of Task 8's JSON
    fingerprint rather than an incidental presentation detail.
    """
    normalized = [
        {
            "source": record.source,
            "source_image_id": record.source_image_id,
            "file_path": record.file_path,
            "width": record.width,
            "height": record.height,
            "split": record.split,
            "label_status": record.label_status,
        }
        for record in records
    ]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sealed_fingerprint(value: object) -> str:
    if not isinstance(value, str) or not _CANONICAL_SHA256.fullmatch(value):
        raise _problem("split manifest unlabeled fingerprint is missing or noncanonical", "fingerprints.unlabeled is absent, not a lowercase SHA-256 digest, or has an unsupported representation", "use the unmodified Task 8 split_manifest.json with its canonical lowercase fingerprints.unlabeled")
    return value


@dataclass(frozen=True)
class SealedUnlabeledMembership:
    """Explicit immutable permission for the only images pseudo generation may read.

    Instances are created from the paired Task 8 manifests, or deliberately by
    a caller that has already sealed an equivalent membership set.  The
    generator intentionally refuses a bare sequence of image records.
    """

    records: tuple[UnlabeledImageRecord, ...]
    protected_image_ids: frozenset[str] = frozenset()
    unlabeled_fingerprint: str | None = None

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, UnlabeledImageRecord) for record in records):
            raise _problem("sealed membership contains an invalid record", "membership was not built from explicit unlabeled image records", "use load_unlabeled_manifest or UnlabeledImageRecord values only")
        source_ids = [record.source_image_id for record in records]
        if len(set(source_ids)) != len(source_ids):
            raise _problem("sealed membership duplicates source_image_id", "two rows would produce indistinguishable pseudo-label provenance", "deduplicate source_image_id values before sealing membership")
        protected = frozenset(self.protected_image_ids)
        if any(not isinstance(source_id, str) or not source_id.strip() for source_id in protected):
            raise _problem("sealed membership has an invalid protected image ID", "a protected split did not contain a nonempty image identifier", "regenerate the Task 8 split manifest")
        overlap = sorted(set(source_ids) & protected)
        if overlap:
            raise _problem("sealed unlabeled membership contains protected image IDs", f"the records overlap protected IDs {overlap!r}", "remove validation, test, pseudo-audit, and external-test images from pseudo generation")
        recomputed = canonical_unlabeled_fingerprint(records)
        expected = _sealed_fingerprint(self.unlabeled_fingerprint)
        if not hmac.compare_digest(recomputed, expected):
            raise _problem("sealed unlabeled membership fingerprint differs from Task 8", "the image source, ID, path, dimensions, split, status, or record order changed after split creation", "use the unmodified paired unlabeled.json and split_manifest.json outputs")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "protected_image_ids", protected)
        object.__setattr__(self, "unlabeled_fingerprint", expected)


def _load_task8_membership(unlabeled_path: Path, split_manifest_path: Path) -> SealedUnlabeledMembership:
    try:
        payload = json.loads(unlabeled_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("unlabeled manifest cannot be read as JSON", str(error), "provide a readable UTF-8 Task 8 unlabeled.json file") from error
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise _problem("unlabeled manifest has no records array", "the input is not the explicit Task 8 unlabeled image manifest", "provide {'records': [...]} from unlabeled.json, not a labels or annotations manifest")
    safe_records = tuple(_safe_manifest_row(row) for row in records)
    try:
        split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("split manifest cannot be read as JSON", str(error), "supply the paired readable Task 8 split_manifest.json") from error
    if not isinstance(split_payload, Mapping):
        raise _problem("split manifest is not an object", "the paired Task 8 split manifest is malformed", "regenerate split_manifest.json with create_splits")
    sealed_ids = _id_set(split_payload.get("unlabeled_image_ids"), field="unlabeled_image_ids")
    record_ids = frozenset(record.source_image_id for record in safe_records)
    if record_ids != sealed_ids:
        raise _problem("unlabeled manifest membership differs from split manifest", "pseudo inputs are not the exact Task 8 unlabeled membership", "use the unmodified paired unlabeled.json and split_manifest.json outputs")
    fingerprints = split_payload.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise _problem("split manifest has no fingerprints object", "the paired Task 8 split manifest cannot seal the unlabeled artifact", "regenerate split_manifest.json with create_splits")
    expected_fingerprint = _sealed_fingerprint(fingerprints.get("unlabeled"))
    split_ids = split_payload.get("split_image_ids")
    if not isinstance(split_ids, Mapping):
        raise _problem("split manifest has no split_image_ids object", "the paired Task 8 split manifest is malformed", "regenerate split_manifest.json with create_splits")
    expected_protected = {"validation", "test", "pseudo_audit", "external_test"}
    if set(split_ids) != expected_protected:
        raise _problem("split manifest protected split keys are invalid", "the paired Task 8 split manifest does not seal every protected partition", "regenerate split_manifest.json with create_splits")
    protected_ids = frozenset().union(*(_id_set(split_ids[name], field=f"split_image_ids.{name}") for name in expected_protected))
    return SealedUnlabeledMembership(safe_records, protected_ids, expected_fingerprint)


def load_unlabeled_manifest(path: Path, *, split_manifest_path: Path | None = None) -> SealedUnlabeledMembership:
    """Load only the paired Task 8 unlabeled and split-membership manifests.

    ``split_manifest_path`` defaults to ``split_manifest.json`` beside the
    supplied ``unlabeled.json``.  This deliberately makes a copied or edited
    unlabeled list insufficient authority to access images.
    """
    manifest = path.resolve(strict=False)
    split_path = (split_manifest_path or manifest.with_name("split_manifest.json")).resolve(strict=False)
    return _load_task8_membership(manifest, split_path)


def _default_image_loader(path: Path) -> object:
    try:
        from PIL import Image

        with Image.open(path) as opened:
            opened.load()
            return opened.copy()
    except (OSError, ValueError) as error:
        raise _problem("unlabeled image cannot be decoded", str(error), "restore a Pillow-readable source image or remove it from the unlabeled manifest") from error


def _size_if_available(image: object) -> tuple[int, int] | None:
    value = getattr(image, "size", None)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    width, height = value
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in (width, height)):
        return None
    return width, height


def _validate_original_box(detection: DetectionRecord, *, width: int, height: int) -> tuple[float, float, float, float]:
    """Bound raw original-view predictions before they enter future trust filtering."""
    x1, y1, x2, y2 = detection.xyxy
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise _problem("detector box lies outside the unlabeled image", "teacher output and manifest dimensions are inconsistent", "use a teacher whose output coordinates match the original input resolution")
    return detection.xyxy


@dataclass(frozen=True)
class PseudoGenerationResult:
    """Unfiltered candidates retained in deterministic input/view/prediction order."""

    teacher_run_id: str
    candidates: tuple[PseudoCandidate, ...]

    def __post_init__(self) -> None:
        teacher_run_id = _text(self.teacher_run_id, "teacher_run_id")
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, PseudoCandidate) for candidate in candidates):
            raise _problem("pseudo generation result contains an invalid candidate", "candidate publication was given an unvalidated record", "retain only PseudoCandidate objects returned by PseudoLabelGenerator")
        mismatched = [candidate.source_image_id for candidate in candidates if candidate.teacher_run_id != teacher_run_id]
        if mismatched:
            raise _problem("pseudo generation result mixes teacher run IDs", f"candidates for {mismatched!r} do not match {teacher_run_id!r}", "publish candidates from one completed teacher run at a time")
        object.__setattr__(self, "teacher_run_id", teacher_run_id)
        object.__setattr__(self, "candidates", candidates)

    def mapping(self) -> dict[str, Any]:
        return {
            "manifest_version": "1.0",
            "teacher_run_id": self.teacher_run_id,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.mapping() for candidate in self.candidates],
        }


class PseudoLabelGenerator:
    """Generate two-view raw candidates while remaining unable to read labels."""

    def __init__(
        self,
        detector: DetectorAdapter,
        *,
        teacher_run_id: str,
        confidence: float | None = None,
        image_root: Path | None = None,
        image_loader: Callable[[Path], object] | None = None,
    ) -> None:
        if not isinstance(detector, DetectorAdapter):
            raise _problem("detector does not implement the project adapter", "a raw backend could bypass the fixed five-class output validation", "pass a DetectorAdapter implementation such as UltralyticsDetectorAdapter")
        self._detector = detector
        self._teacher_run_id = _text(teacher_run_id, "teacher_run_id")
        self._confidence = confidence
        if image_root is None:
            raise _problem("image_root is required", "a pseudo-label run without a sealed image root could read arbitrary absolute paths", "provide the Task 8 dataset image root")
        self._image_root = image_root.resolve(strict=False)
        self._image_loader = image_loader or _default_image_loader

    def _path_for(self, record: UnlabeledImageRecord) -> Path:
        path = Path(record.file_path)
        if path.is_absolute():
            raise _problem("unlabeled image path is absolute", "absolute paths can bypass the sealed image root", "store a relative file_path beneath --image-root")
        if ".." in path.parts:
            raise _problem("unlabeled image path traverses upward", "a relative path can escape the sealed image root", "store a normalized relative file_path beneath --image-root")
        unresolved = self._image_root / path
        # A lexical relative path can otherwise be a symlink to a protected
        # image or artifact *inside* image_root.  Reject every redirection, not
        # only upward traversal, so the membership manifest is the sole image
        # authority.
        component = self._image_root
        for part in path.parts:
            component = component / part
            if component.is_symlink():
                raise _problem("unlabeled image path redirects through a symbolic link", f"{record.file_path!r} is a legal relative path but redirects within the sealed image root", "store a regular source image directly beneath --image-root and regenerate the sealed manifest")
        candidate = unresolved.resolve(strict=False)
        try:
            candidate.relative_to(self._image_root)
        except ValueError as error:
            raise _problem("unlabeled image path escapes image_root", f"{record.file_path!r} resolves outside {self._image_root}", "store a normalized relative file_path beneath --image-root") from error
        return candidate

    def _predict(self, image: object) -> tuple[DetectionRecord, ...]:
        try:
            records = self._detector.predict(image, confidence=self._confidence)
        except DetectorAdapterError as error:
            raise _problem("teacher inference failed", str(error), "verify the teacher checkpoint, adapter, and source image") from error
        if not isinstance(records, tuple) or any(not isinstance(record, DetectionRecord) for record in records):
            raise _problem("detector returned an invalid prediction collection", "the adapter did not return a tuple of validated DetectionRecord objects", "use the project DetectorAdapter interface without bypassing its validation")
        return records

    def generate(self, membership: SealedUnlabeledMembership) -> PseudoGenerationResult:
        """Run original and flip views; mapping flip coordinates back to originals."""
        if not isinstance(membership, SealedUnlabeledMembership):
            raise _problem("generator input is not sealed unlabeled membership", "a labeled annotation or unvalidated record sequence was supplied", "call load_unlabeled_manifest or explicitly construct SealedUnlabeledMembership")
        provided = membership.records
        seen: set[str] = set()
        candidates: list[PseudoCandidate] = []
        for record in provided:
            if record.source_image_id in seen:
                raise _problem("unlabeled inputs duplicate source_image_id", "provenance would not uniquely identify the source image", "deduplicate the explicit unlabeled manifest before generation")
            seen.add(record.source_image_id)
            path = self._path_for(record)
            image = self._image_loader(path)
            decoded_size = _size_if_available(image)
            if decoded_size != (record.width, record.height):
                detail = "no valid dimensions" if decoded_size is None else f"{decoded_size[0]}x{decoded_size[1]}"
                raise _problem("unlabeled image dimensions differ from its manifest", f"manifest is {record.width}x{record.height} but decoded image is {detail}", "regenerate the unlabeled manifest after correcting the image source")
            try:
                flipped = horizontal_flip_image(image)
            except TransformError as error:
                raise _problem("unlabeled image cannot be transformed", str(error), "provide a Pillow-readable image and retry") from error
            flipped_size = _size_if_available(flipped)
            if flipped_size != (record.width, record.height):
                detail = "no valid dimensions" if flipped_size is None else f"{flipped_size[0]}x{flipped_size[1]}"
                raise _problem("horizontal-flip image dimensions differ from its manifest", f"manifest is {record.width}x{record.height} but transformed image is {detail}", "use a dimension-preserving horizontal flip before teacher inference")

            for detection in self._predict(image):
                original = _validate_original_box(detection, width=record.width, height=record.height)
                candidates.append(PseudoCandidate.from_detection(detection, teacher_run_id=self._teacher_run_id, source_image_id=record.source_image_id, source_file_path=record.file_path, view="original", xyxy=original))
            for detection in self._predict(flipped):
                try:
                    original = horizontal_flip_xyxy(detection.xyxy, width=record.width)
                    _validate_original_box(
                        DetectionRecord(class_id=detection.class_id, class_name=detection.class_name, confidence=detection.confidence, xyxy=original, is_unknown=False, source_model=detection.source_model),
                        width=record.width,
                        height=record.height,
                    )
                except (TransformError, ValueError) as error:
                    raise _problem("horizontal-flip prediction cannot map to original coordinates", str(error), "ensure teacher boxes use the same unscaled image dimensions as the manifest") from error
                candidates.append(PseudoCandidate.from_detection(detection, teacher_run_id=self._teacher_run_id, source_image_id=record.source_image_id, source_file_path=record.file_path, view="horizontal_flip", xyxy=original))
        return PseudoGenerationResult(self._teacher_run_id, tuple(candidates))


def write_pseudo_candidates(result: PseudoGenerationResult, output_path: Path) -> Path:
    """Publish a JSON candidate manifest without overwriting an existing artifact."""
    if not isinstance(result, PseudoGenerationResult):
        raise _problem("pseudo candidate output is not a validated generation result", "publication was given a hand-built mapping or candidate list", "pass the PseudoGenerationResult returned by PseudoLabelGenerator")
    # Reconstruct at the publication boundary so a caller cannot make a valid
    # result and then mutate it through a low-level dataclass escape hatch.
    result = PseudoGenerationResult(result.teacher_run_id, result.candidates)
    destination = output_path.resolve(strict=False)
    if destination.exists():
        raise _problem("pseudo candidate output already exists", f"{destination} would be overwritten", "choose a new output path or archive the prior candidate artifact")
    ancestor = destination.parent
    while ancestor != ancestor.parent:
        if ancestor.exists() and not ancestor.is_dir():
            raise _problem("pseudo candidate output has a file ancestor", f"{ancestor} is not a directory", "choose an output path beneath existing directories")
        ancestor = ancestor.parent
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(result.mapping(), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        # A hard-link publish is exclusive: another process cannot replace an
        # existing artifact between our first existence check and publication.
        os.link(temporary, destination)
        temporary.unlink()
        return destination
    except FileExistsError as error:
        raise _problem("pseudo candidate output already exists", f"{destination} was created by another process", "choose a new output path; existing candidate evidence is never replaced") from error
    except OSError as error:
        raise _problem("pseudo candidate output could not be written", str(error), "ensure the output parent is writable and choose a new output path") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
