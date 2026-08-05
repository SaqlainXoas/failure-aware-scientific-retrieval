from unittest.mock import patch

import pytest

from retrieval.data import resolve_device


def test_resolve_device_cpu_passthrough():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_unknown_raises():
    with pytest.raises(ValueError):
        resolve_device("cuda")


@patch("torch.backends.mps.is_available", return_value=True)
def test_resolve_device_auto_prefers_mps_when_available(_mock):
    assert resolve_device("auto") == "mps"


@patch("torch.backends.mps.is_available", return_value=False)
def test_resolve_device_auto_falls_back_to_cpu(_mock):
    assert resolve_device("auto") == "cpu"


@patch("torch.backends.mps.is_available", return_value=True)
def test_resolve_device_mps_when_available(_mock):
    assert resolve_device("mps") == "mps"


@patch("torch.backends.mps.is_available", return_value=False)
def test_resolve_device_mps_requested_but_unavailable_raises(_mock):
    with pytest.raises(RuntimeError):
        resolve_device("mps")
