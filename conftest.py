"""
Root pytest configuration.

Markers:
    slow        — long-running tests, skipped by default in CI (use -m slow)
    requires_r  — tests that need R + rpy2 + SILGGM; skipped automatically
                  when the rpy2 import fails
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line(
        "markers",
        "requires_r: mark test as requiring R, rpy2, and SILGGM",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip requires_r tests when rpy2 is unavailable."""
    try:
        import rpy2  # noqa: F401
    except ImportError:
        skip_r = pytest.mark.skip(reason="rpy2 not installed — skipping R parity tests")
        for item in items:
            if "requires_r" in item.keywords:
                item.add_marker(skip_r)
