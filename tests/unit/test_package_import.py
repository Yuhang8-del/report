"""Unit tests for the public package identity."""

from __future__ import annotations


def test_package_exposes_stable_identity() -> None:
    """The package can be imported before any optional application setup."""
    import fruit_ssod

    assert fruit_ssod.__version__ == "0.1.0"
    assert fruit_ssod.PACKAGE_NAME == "fruit-ssod"
