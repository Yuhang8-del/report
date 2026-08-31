"""Contract tests for the customer-authorized post-Student discovery stage."""

from pathlib import Path

import numpy as np

from fruit_ssod.open_world.discovery import NOVEL_CLASSES, _kmeans, evaluate_clusters


def test_novel_registry_excludes_the_known_five_classes() -> None:
    assert NOVEL_CLASSES == ("Avocado", "Blueberry", "Cherry", "Kiwi", "Mango", "Rockmelon")
    assert not set(NOVEL_CLASSES).intersection({"Apple", "Banana", "Orange", "Strawberry", "Pineapple"})


def test_kmeans_is_deterministic_for_fixed_seed() -> None:
    features = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]],
        dtype=np.float32,
    )
    first_centers, first_labels = _kmeans(features, 2, seed=42)
    second_centers, second_labels = _kmeans(features, 2, seed=42)
    assert np.array_equal(first_labels, second_labels)
    assert np.allclose(first_centers, second_centers)


def test_cluster_metrics_are_perfect_for_matching_clusters() -> None:
    metrics = evaluate_clusters([0, 0, 1, 1, 2, 2], ["Mango", "Mango", "Kiwi", "Kiwi", "Cherry", "Cherry"])
    assert metrics["purity"] == 1.0
    assert metrics["nmi"] == 1.0
    assert metrics["ari"] == 1.0
    assert metrics["posthoc_cluster_to_category"] == {"0": "Mango", "1": "Kiwi", "2": "Cherry"}
