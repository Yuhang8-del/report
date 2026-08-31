"""Local-only importer wrapper for the reviewed Hugging Face crop/plant archive."""

from __future__ import annotations

from pathlib import Path

from fruit_ssod.data.kaggle_fruit_detection import KaggleFruitDetectionImportResult, import_yolo_fruit_detection
from fruit_ssod.data.schema import LicenseMetadata


SOURCE_NAME = "hf_crop_plant_25k"


def import_hf_crop_plant(
    dataset_root: Path,
    data_yaml: Path,
    *,
    source_version: str,
    source_page: str,
    license_metadata: LicenseMetadata,
) -> KaggleFruitDetectionImportResult:
    """Index reviewed YOLO boxes without changing source partitions or labels."""
    return import_yolo_fruit_detection(
        dataset_root,
        data_yaml,
        source_name=SOURCE_NAME,
        source_version=source_version,
        source_page=source_page,
        license_metadata=license_metadata,
        allow_polygons=True,
    )
