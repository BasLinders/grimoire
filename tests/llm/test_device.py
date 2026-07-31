"""Tests for grimoire_ai.llm.device.select_device.

Verifies the CUDA > MPS > CPU preference order and that an explicit
``preferred`` override always wins over auto-detection.
"""

from unittest.mock import patch

from grimoire_ai.llm.device import select_device


def test_explicit_preference_wins():
    assert select_device("cpu") == "cpu"
    assert select_device("cuda") == "cuda"
    assert select_device("mps") == "mps"


def test_prefers_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.backends.mps.is_available", return_value=True):
        assert select_device() == "cuda"


def test_falls_back_to_mps_on_apple_silicon():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True):
        assert select_device() == "mps"


def test_falls_back_to_cpu_when_no_gpu():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        assert select_device() == "cpu"
