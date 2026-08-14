"""Tests for the Pre-train tab's attention-type / MLA field wiring.

MLA (docs/architecture_optimization.md item #1) is exposed as a "gqa"/"mla"
dropdown in the Model architecture accordion, with two MLA-only dimension
fields shown only when "mla" is selected. These tests cover the two pure
helpers behind that wiring; the actual training launch (``run_pretrain``) is
not unit tested here, consistent with the rest of this test module — see
test_app_ft_steps.py for the same scoping choice on the fine-tune tab.
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from grimoire_ai.ui import app  # noqa: E402


def test_toggle_mla_fields_hidden_for_gqa():
    update = app._toggle_mla_fields("gqa")
    assert update["visible"] is False


def test_toggle_mla_fields_visible_for_mla():
    update = app._toggle_mla_fields("mla")
    assert update["visible"] is True


def test_mla_dim_zero_means_auto():
    """The UI's 0-means-auto sentinel must translate to TransformerConfig's None."""
    assert app._mla_dim_or_none(0) is None
    assert app._mla_dim_or_none(0.0) is None


def test_mla_dim_nonzero_passes_through_as_int():
    assert app._mla_dim_or_none(128.0) == 128
    assert isinstance(app._mla_dim_or_none(128.0), int)
