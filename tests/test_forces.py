"""Regression tests for the Phase-3 non-conservative force field."""
import numpy as np
import pytest

from params import NonConservativeParams
from forces import (
    conservative_force,
    scattering_force,
    total_force,
    curl_theta_analytic,
    curl_theta_numeric,
)


def test_w0_defaults_to_sigma_r():
    p = NonConservativeParams()
    assert p.w0 == pytest.approx(p.sigma_r)


def test_zero_F0_matches_conservative():
    p = NonConservativeParams(F0=0.0)
    r = np.linspace(0.0, 5 * p.sigma_r, 50)
    z = np.linspace(-3 * p.sigma_z, 3 * p.sigma_z, 50)
    Fr_c, Fz_c = conservative_force(p, r, z)
    Fr_t, Fz_t = total_force(p, r, z)
    assert np.allclose(Fr_t, Fr_c)
    assert np.allclose(Fz_t, Fz_c)


def test_zero_F0_zero_scattering_force():
    p = NonConservativeParams(F0=0.0)
    r = np.linspace(0.0, 5 * p.sigma_r, 50)
    assert np.allclose(scattering_force(p, r), 0.0)


def test_zero_F0_zero_curl():
    p = NonConservativeParams(F0=0.0)
    r = np.linspace(0.0, 5 * p.sigma_r, 50)
    assert np.allclose(curl_theta_analytic(p, r), 0.0)


def test_nonzero_F0_nonzero_curl_off_axis():
    p = NonConservativeParams(F0=1e-14)
    r = np.linspace(0.1 * p.sigma_r, 5 * p.sigma_r, 50)
    curl = curl_theta_analytic(p, r)
    assert np.all(curl > 0), "curl should be strictly positive off-axis for F0 > 0"


def test_curl_zero_on_axis():
    p = NonConservativeParams(F0=1e-14)
    assert curl_theta_analytic(p, np.array([0.0]))[0] == 0.0


def test_curl_analytic_matches_numeric():
    p = NonConservativeParams(F0=1e-14)
    r = np.linspace(0.0, 6 * p.sigma_r, 2000)
    analytic = curl_theta_analytic(p, r)[1:-1]
    numeric = curl_theta_numeric(p, r)
    rel_err = np.max(np.abs(numeric - analytic)) / np.max(np.abs(analytic))
    assert rel_err < 1e-4, f"analytic/numeric curl mismatch: {rel_err:.2e}"
