"""
Unit tests for IOHMM forecast methods.

Covers:
  - predict_state output sums to 1
  - t+2 has higher entropy than t+1 (more diffuse further ahead)
  - predict_state(steps=2) == chained predict_state(steps=1) twice
  - predict_obs returns weighted emission mean
"""

from __future__ import annotations

import numpy as np
import pytest

from schism.models.iohmm import IOHMM


def _make_model() -> IOHMM:
    """K=2, D=2, M=1 IOHMM with hand-set parameters. No EM needed."""
    m = IOHMM(n_states=2, n_obs=2, n_exog=1)
    m.pi    = np.array([0.6, 0.4])
    m.mu    = np.array([[0.0, 0.0], [1.0, 1.0]])
    m.sigma = np.array([np.eye(2), np.eye(2)])
    m.alpha = np.array([[1.0, -1.0], [-1.0, 1.0]])  # sticky self-transition
    m.beta  = np.zeros((2, 2, 1))
    m.tau   = 2.0
    m.labels    = ["low", "high"]
    m.model_ver = "test"
    m._fitted   = True
    return m


class TestPredictState:
    def test_output_sums_to_one(self):
        m = _make_model()
        f1 = m.predict_state(np.array([0.8, 0.2]), steps=1)
        assert abs(f1.sum() - 1.0) < 1e-10

    def test_t2_more_diffuse_than_t1(self):
        m = _make_model()
        gamma = np.array([0.95, 0.05])
        f1 = m.predict_state(gamma, steps=1)
        f2 = m.predict_state(gamma, steps=2)
        entropy = lambda p: -np.sum(p * np.log(p + 1e-10))
        assert entropy(f2) >= entropy(f1)

    def test_steps2_equals_chained_steps1(self):
        m = _make_model()
        gamma = np.array([0.7, 0.3])
        f2_direct  = m.predict_state(gamma, steps=2)
        f2_chained = m.predict_state(m.predict_state(gamma, steps=1), steps=1)
        np.testing.assert_allclose(f2_direct, f2_chained, atol=1e-12)

    def test_u_none_uses_zeros(self):
        m = _make_model()
        gamma = np.array([0.6, 0.4])
        f_none  = m.predict_state(gamma, u_next=None, steps=1)
        f_zeros = m.predict_state(gamma, u_next=np.zeros(1), steps=1)
        np.testing.assert_allclose(f_none, f_zeros, atol=1e-12)


class TestPredictObs:
    def test_shape(self):
        m = _make_model()
        obs = m.predict_obs(np.array([0.6, 0.4]))
        assert obs.shape == (2,)

    def test_weighted_mean(self):
        m = _make_model()
        # mu[0]=[0,0], mu[1]=[1,1]; weights 50/50 → expected [0.5, 0.5]
        obs = m.predict_obs(np.array([0.5, 0.5]))
        np.testing.assert_allclose(obs, [0.5, 0.5])

    def test_pure_state_returns_that_state_mean(self):
        m = _make_model()
        np.testing.assert_allclose(m.predict_obs(np.array([1.0, 0.0])), m.mu[0])
        np.testing.assert_allclose(m.predict_obs(np.array([0.0, 1.0])), m.mu[1])
