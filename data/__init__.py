"""Public canonical data schema and class-mapping API.

Author: Fruit SSOD contributors
Date: 2026-07-31
Version: 1.0.0
"""

from fruit_ssod.data.class_mapping import (
    CanonicalClass,
    ClassMappingError,
    ClassRegistry,
    DEFAULT_CLASS_REGISTRY,
    load_class_registry,
    resolve_class_id,
)
from fruit_ssod.data.schema import (
    ALLOWED_LABEL_STATUSES,
    ALLOWED_OBJECT_SPLIT_STATUSES,
    ALLOWED_SPLITS,
    ALLOWED_UNLABELED_IMAGE_SPLIT_STATUSES,
    AnnotationValidationError,
    CanonicalAnnotation,
    LicenseMetadata,
    UnlabeledImageRecord,
)

__all__ = [
    "ALLOWED_LABEL_STATUSES",
    "ALLOWED_OBJECT_SPLIT_STATUSES",
    "ALLOWED_SPLITS",
    "ALLOWED_UNLABELED_IMAGE_SPLIT_STATUSES",
    "AnnotationValidationError",
    "CanonicalAnnotation",
    "CanonicalClass",
    "ClassMappingError",
    "ClassRegistry",
    "DEFAULT_CLASS_REGISTRY",
    "LicenseMetadata",
    "UnlabeledImageRecord",
    "load_class_registry",
    "resolve_class_id",
]
