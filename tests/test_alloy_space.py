"""Tests for alloy_space composition and strength model modules."""

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pytest

from alloy_space import generate_compositions, model_control
from alloy_space.strength_model import model, temp_model


class TestGenerateCompositions(TestCase):
    """Tests for generate_compositions."""

    def test_rows_sum_to_one(self):
        """All rows sum to 1.0 within floating-point tolerance."""
        df = generate_compositions(["A", "B", "C"], n_comps=5, max_order=3)
        assert np.allclose(df.sum(axis=1), 1.0)

    def test_no_negative_fractions(self):
        """All fractions are non-negative."""
        df = generate_compositions(["A", "B", "C"], n_comps=5)
        assert (df.values >= 0).all()

    def test_no_duplicates(self):
        """No duplicate rows in output."""
        df = generate_compositions(["A", "B", "C", "D"], n_comps=5, max_order=3)
        assert not df.duplicated().any()

    def test_invalid_max_order_raises(self):
        """max_order < 2 raises ValueError."""
        with pytest.raises(ValueError):
            generate_compositions(["A", "B"], n_comps=5, max_order=1)

    def test_max_order_exceeded_raises(self):
        """max_order > len(elements) raises ValueError."""
        with pytest.raises(ValueError):
            generate_compositions(["A", "B"], n_comps=5, max_order=5)


class TestStrengthModel(TestCase):
    """Tests for the Maresca-Curtin BCC strength model."""

    def setUp(self):
        """Mo-Nb-Ta-W equimolar refractory HEA."""
        self.composition = {"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25}

    def test_tau_y0_positive(self):
        """Athermal yield stress is positive for a real alloy."""
        result = model(self.composition)
        assert result["tau_y_0"] > 0

    def test_delta_eb_positive(self):
        """Activation barrier is positive."""
        result = model(self.composition)
        assert result["delta_Eb"] > 0

    def test_misfit_keys_match_elements(self):
        """Misfit dict contains exactly the input element keys."""
        result = model(self.composition)
        assert set(result["misfit"].keys()) == set(self.composition.keys())

    def test_temp_model_decreases_with_temperature(self):
        """Flow stress decreases monotonically with temperature."""
        result = model(self.composition)
        temps = np.array([300.0, 600.0, 900.0, 1200.0, 1500.0])
        tau = temp_model(result, eps_dot=0.001, temperature=temps)
        assert np.all(np.diff(tau) < 0)

    def test_model_control_returns_tuple(self):
        """model_control returns a 3-tuple of floats."""
        out = model_control(["Mo", "Nb", "Ta", "W"], [0.25, 0.25, 0.25, 0.25])
        assert len(out) == 3
        assert all(isinstance(v, float) for v in out)

    def test_unknown_element_raises(self):
        """Unknown element raises KeyError."""
        with pytest.raises(KeyError):
            model({"Unobtainium": 1.0})
