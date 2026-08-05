"""Shared fixtures.

Integration tests need the real SciFact dataset. They run as part of the default
`uv run pytest` — plan §16 lists the split/leakage checks as required unit tests, so
excluding them from the documented command would leave the strongest leakage guard
unexercised. When the dataset genuinely cannot be reached they skip with a clear
message rather than failing an otherwise-valid checkout.
"""

import pytest


def _dataset_unavailable(exc: Exception) -> bool:
    """Only network/cache failures may skip integrity tests; code defects must still fail."""
    return isinstance(exc, (OSError, ConnectionError, TimeoutError)) or (
        isinstance(exc, RuntimeError)
        and str(exc).startswith("All download sources failed")
    )


@pytest.fixture(scope="session")
def scifact():
    """Loads both SciFact splits once per session, or skips the test that asked for it."""
    from retrieval.data import load_scifact_split

    try:
        return {"train": load_scifact_split("train"), "test": load_scifact_split("test")}
    except Exception as exc:
        if not _dataset_unavailable(exc):
            raise
        pytest.skip(f"SciFact unavailable via ir_datasets ({exc.__class__.__name__}: {exc})")
