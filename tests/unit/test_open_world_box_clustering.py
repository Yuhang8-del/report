from __future__ import annotations

import numpy as np

from fruit_ssod.open_world.box_clustering import normalize_features


def test_normalize_features_uses_frozen_discovery_mean() -> None:
    features = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    mean = np.array([[1.0, 1.0]], dtype=np.float32)
    normalized = normalize_features(features, mean)
    np.testing.assert_allclose(normalized[0], [1.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])
