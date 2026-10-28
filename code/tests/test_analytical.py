"""Tests for mmc_analytical: every identity in the consistency checklist plus error handling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmc_analytical import (
    MM2Theoretical,
    P_no_wait_mm2,
    P_Nq_ge_K_mm2,
    P_N_ge_Nstar_mm2,
    P_wait_gt_t_general,
    P_wait_gt_t_mm2,
    L_mm2,
    L_q_mm2,
    L_q_general,
    W_mm2,
    W_q_mm2,
    W_q_general,
    all_metrics_mm2,
    conditional_wait_rate_mm2,
    erlang_c_general,
    erlang_c_mm2,
    expected_busy_servers_mm2,
    offered_load,
    pi0_mm2,
    pi_n_mm2,
    pi_vector_mm2,
    rho_from_rates,
    truncation_n_max,
    utilization_mm2,
    waiting_time_cdf_mm2,
)

RHOS = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
MU = 1.0
TOL = 1e-12


# Identity 1 - Normalization

@pytest.mark.parametrize("rho", RHOS)
def test_identity_1_normalization(rho):
    n_max = truncation_n_max(rho, tol=1e-14)
    pi = pi_vector_mm2(rho, n_max=n_max)
    assert pi.sum() == pytest.approx(1.0, abs=TOL)


# Identity 2 - Uniform tail form for n >= 1

@pytest.mark.parametrize("rho", RHOS)
def test_identity_2_uniform_tail(rho):
    pi0 = pi0_mm2(rho)
    for n in range(1, 30):
        expected = 2.0 * (rho ** n) * pi0
        assert pi_n_mm2(rho, n) == pytest.approx(expected, abs=TOL, rel=1e-13)


# Identity 3 - Geometric ratio on the tail

@pytest.mark.parametrize("rho", RHOS)
def test_identity_3_geometric_ratio(rho):
    for n in range(1, 30):
        ratio = pi_n_mm2(rho, n + 1) / pi_n_mm2(rho, n)
        assert ratio == pytest.approx(rho, abs=TOL)


# Identity 4 - Erlang C as both summation and closed form

@pytest.mark.parametrize("rho", RHOS)
def test_identity_4_erlang_c(rho):
    closed = erlang_c_mm2(rho)
    n_max = truncation_n_max(rho, tol=1e-14)
    pi = pi_vector_mm2(rho, n_max=n_max)
    summed = pi[2:].sum()
    expected = 2.0 * rho ** 2 / (1.0 + rho)
    assert closed == pytest.approx(expected, abs=TOL)
    # Sum of pi tail should match analytical Erlang C to machine precision
    assert summed == pytest.approx(closed, abs=1e-12)


# Identity 5 - P(no wait) = pi_0 + pi_1

@pytest.mark.parametrize("rho", RHOS)
def test_identity_5_no_wait(rho):
    pi0 = pi0_mm2(rho)
    pi1 = pi_n_mm2(rho, 1)
    expected = (1.0 - rho) * (1.0 + 2.0 * rho) / (1.0 + rho)
    assert pi0 + pi1 == pytest.approx(expected, abs=TOL)
    assert P_no_wait_mm2(rho) == pytest.approx(expected, abs=TOL)
    assert 1.0 - erlang_c_mm2(rho) == pytest.approx(expected, abs=TOL)


# Identity 6 - L decomposition: L - L_q = E[busy] = 2 rho

@pytest.mark.parametrize("rho", RHOS)
def test_identity_6_decomposition(rho):
    diff = L_mm2(rho) - L_q_mm2(rho)
    assert diff == pytest.approx(2.0 * rho, abs=1e-12, rel=1e-12)
    assert expected_busy_servers_mm2(rho) == pytest.approx(2.0 * rho, abs=TOL)


# Identity 7 - Closed-form L and L_q values match recursion

@pytest.mark.parametrize("rho", RHOS)
def test_identity_7_closed_form_L(rho):
    # L involves n * pi_n which decays slower than pi_n, need more terms
    n_max = max(truncation_n_max(rho, tol=1e-16), 1000)
    pi = pi_vector_mm2(rho, n_max=n_max)
    ns = np.arange(pi.size)
    L_via_sum = float((ns * pi).sum())
    L_q_via_sum = float((np.maximum(ns - 2, 0) * pi).sum())
    assert L_mm2(rho) == pytest.approx(L_via_sum, abs=1e-9, rel=1e-9)
    assert L_q_mm2(rho) == pytest.approx(L_q_via_sum, abs=1e-9, rel=1e-9)


# Identity 8 - Little's Law on queue and on system

@pytest.mark.parametrize("rho", RHOS)
def test_identity_8_littles_law(rho):
    lam = 2.0 * rho * MU
    assert W_q_mm2(rho, MU) == pytest.approx(L_q_mm2(rho) / lam, abs=1e-12, rel=1e-12)
    assert W_mm2(rho, MU) == pytest.approx(L_mm2(rho) / lam, abs=1e-12, rel=1e-12)


# Identity 9 - Sojourn decomposition: W - W_q = 1/mu

@pytest.mark.parametrize("rho", RHOS)
def test_identity_9_sojourn_decomposition(rho):
    diff = W_mm2(rho, MU) - W_q_mm2(rho, MU)
    assert diff == pytest.approx(1.0 / MU, abs=1e-12)


# Identity 10 - Utilization identity

@pytest.mark.parametrize("rho", RHOS)
def test_identity_10_utilization(rho):
    assert utilization_mm2(rho) == pytest.approx(rho, abs=TOL)
    assert expected_busy_servers_mm2(rho) == pytest.approx(2.0 * rho, abs=TOL)


# Identity 11 - Wait tail at zero equals Erlang C

@pytest.mark.parametrize("rho", RHOS)
def test_identity_11_wait_tail_at_zero(rho):
    assert P_wait_gt_t_mm2(rho, MU, 0.0) == pytest.approx(erlang_c_mm2(rho), abs=TOL)
    assert waiting_time_cdf_mm2(rho, MU, 0.0) == pytest.approx(
        1.0 - erlang_c_mm2(rho), abs=TOL
    )


# Identity 12 - Strict monotonicity of P(W_q > t)

@pytest.mark.parametrize("rho", RHOS)
def test_identity_12_wait_tail_monotone(rho):
    grid = np.linspace(0.0, 10.0, 50)
    vals = [P_wait_gt_t_mm2(rho, MU, t) for t in grid]
    diffs = np.diff(vals)
    assert (diffs <= 1e-15).all()
    assert vals[0] == pytest.approx(erlang_c_mm2(rho), abs=TOL)
    rate = conditional_wait_rate_mm2(rho, MU)
    assert rate == pytest.approx(2.0 * MU * (1.0 - rho), abs=TOL)


# Identity 13 - Queue tail structure

@pytest.mark.parametrize("rho", RHOS)
def test_identity_13_queue_tail(rho):
    assert P_Nq_ge_K_mm2(rho, 0) == pytest.approx(erlang_c_mm2(rho), abs=TOL)
    for K in range(1, 12):
        ratio = P_Nq_ge_K_mm2(rho, K) / P_Nq_ge_K_mm2(rho, K - 1)
        assert ratio == pytest.approx(rho, abs=TOL)
        closed = 2.0 * (rho ** (K + 2)) / (1.0 + rho)
        assert P_Nq_ge_K_mm2(rho, K) == pytest.approx(closed, abs=TOL)


# Identity 14 - Stress-state formulas

@pytest.mark.parametrize("rho", RHOS)
def test_identity_14_stress_states(rho):
    assert P_N_ge_Nstar_mm2(rho, 0) == pytest.approx(1.0, abs=TOL)
    assert P_N_ge_Nstar_mm2(rho, 1) == pytest.approx(
        1.0 - pi0_mm2(rho), abs=TOL
    )
    assert P_N_ge_Nstar_mm2(rho, 1) == pytest.approx(
        2.0 * rho / (1.0 + rho), abs=TOL
    )
    for Ns in (2, 3, 5, 7, 10):
        closed = 2.0 * (rho ** Ns) / (1.0 + rho)
        assert P_N_ge_Nstar_mm2(rho, Ns) == pytest.approx(closed, abs=TOL)


# Identity 15 - Re-verify Little's Law from closed forms

@pytest.mark.parametrize("rho", RHOS)
def test_identity_15_littles_law_reverification(rho):
    lam = 2.0 * rho * MU
    assert L_mm2(rho) / lam - W_mm2(rho, MU) == pytest.approx(0.0, abs=1e-12)
    assert L_q_mm2(rho) / lam - W_q_mm2(rho, MU) == pytest.approx(0.0, abs=1e-12)


# Identity 16 - Boundary behavior

def test_identity_16_boundary_low():
    rho = 1e-8
    assert pi0_mm2(rho) == pytest.approx(1.0, abs=1e-7)
    assert erlang_c_mm2(rho) < 1e-15
    assert L_q_mm2(rho) < 1e-22
    assert W_q_mm2(rho, MU) < 1e-15
    for t in (0.5, 1.0, 5.0):
        assert P_wait_gt_t_mm2(rho, MU, t) < 1e-15


def test_identity_16_boundary_high():
    # Numerical, monotone push toward 1 from below
    rho_high = 0.999
    L_q = L_q_mm2(rho_high)
    W_q = W_q_mm2(rho_high, MU)
    erl = erlang_c_mm2(rho_high)
    pi0 = pi0_mm2(rho_high)
    assert L_q > 100.0
    assert W_q > 100.0
    assert erl > 0.99
    assert pi0 < 0.001


# Identity 17 - Probability constraints

@pytest.mark.parametrize("rho", RHOS)
def test_identity_17_probability_constraints(rho):
    n_max = truncation_n_max(rho, tol=1e-14)
    pi = pi_vector_mm2(rho, n_max=n_max)
    assert (pi >= 0).all()
    assert pi.sum() == pytest.approx(1.0, abs=TOL)


# Identity 18 - Monotonicity across rho

def test_identity_18_monotone_in_rho():
    grid = np.linspace(0.05, 0.95, 50)
    L_q_vals = [L_q_mm2(r) for r in grid]
    L_vals = [L_mm2(r) for r in grid]
    W_q_vals = [W_q_mm2(r, MU) for r in grid]
    W_vals = [W_mm2(r, MU) for r in grid]
    erl_vals = [erlang_c_mm2(r) for r in grid]
    U_vals = [utilization_mm2(r) for r in grid]
    for vals in (L_q_vals, L_vals, W_q_vals, W_vals, erl_vals, U_vals):
        diffs = np.diff(vals)
        assert (diffs > 0).all(), f"non-monotone series: {vals}"


@pytest.mark.parametrize("rho", RHOS)
def test_identity_18_pi_n_monotone(rho):
    pi = pi_vector_mm2(rho, n_max=50)
    # pi_n is non-increasing in n for n >= 1
    diffs = np.diff(pi[1:])
    assert (diffs <= 1e-15).all()


# Error handling

def test_value_error_at_rho_one():
    with pytest.raises(ValueError):
        pi0_mm2(1.0)
    with pytest.raises(ValueError):
        erlang_c_mm2(1.0)
    with pytest.raises(ValueError):
        L_q_mm2(1.0)
    with pytest.raises(ValueError):
        W_q_mm2(1.0, MU)


def test_value_error_above_one():
    with pytest.raises(ValueError):
        pi0_mm2(1.5)
    with pytest.raises(ValueError):
        erlang_c_mm2(1.2)


def test_value_error_negative():
    with pytest.raises(ValueError):
        pi0_mm2(-0.1)
    with pytest.raises(ValueError):
        pi_n_mm2(0.5, -1)
    with pytest.raises(ValueError):
        P_Nq_ge_K_mm2(0.5, -1)
    with pytest.raises(ValueError):
        P_N_ge_Nstar_mm2(0.5, -1)


def test_rates_helpers():
    assert rho_from_rates(1.5, 1.0) == pytest.approx(0.75, abs=TOL)
    assert offered_load(1.5, 1.0) == pytest.approx(1.5, abs=TOL)
    with pytest.raises(ValueError):
        rho_from_rates(0.0, 1.0)
    with pytest.raises(ValueError):
        offered_load(1.0, 0.0)


# General M/M/c specializes to M/M/2

@pytest.mark.parametrize("rho", RHOS)
def test_general_mmc_matches_mm2_erlang(rho):
    lam = 2.0 * rho * MU
    closed = 2.0 * rho ** 2 / (1.0 + rho)
    via_general = erlang_c_general(2, lam, MU)
    assert via_general == pytest.approx(closed, abs=1e-12)


@pytest.mark.parametrize("rho", RHOS)
def test_general_mmc_matches_mm2_W_q(rho):
    lam = 2.0 * rho * MU
    closed = rho ** 2 / (MU * (1.0 - rho ** 2))
    via_general = W_q_general(2, lam, MU)
    assert via_general == pytest.approx(closed, abs=1e-12, rel=1e-12)


@pytest.mark.parametrize("rho", RHOS)
def test_general_mmc_matches_mm2_L_q(rho):
    lam = 2.0 * rho * MU
    closed = 2.0 * rho ** 3 / (1.0 - rho ** 2)
    via_general = L_q_general(2, lam, MU)
    assert via_general == pytest.approx(closed, abs=1e-12, rel=1e-12)


@pytest.mark.parametrize("rho", RHOS)
def test_general_wait_tail_matches_mm2(rho):
    lam = 2.0 * rho * MU
    for t in (0.0, 1.0, 2.0, 3.0, 5.0):
        closed = (2.0 * rho ** 2 / (1.0 + rho)) * math.exp(-2.0 * MU * (1.0 - rho) * t)
        via_general = P_wait_gt_t_general(2, lam, MU, t)
        assert via_general == pytest.approx(closed, abs=1e-12, rel=1e-12)


def test_mm3_at_known_load():
    # Sanity: at light load M/M/3 should outperform M/M/2.
    lam = 1.0
    mu = 1.0
    C2 = erlang_c_general(2, lam, mu)
    C3 = erlang_c_general(3, lam, mu)
    Wq2 = W_q_general(2, lam, mu)
    Wq3 = W_q_general(3, lam, mu)
    assert C3 < C2
    assert Wq3 < Wq2


# Bundle helper

def test_bundle_consistency():
    bundle = all_metrics_mm2(0.8, MU)
    assert isinstance(bundle, MM2Theoretical)
    assert bundle.rho == 0.8
    assert bundle.lam == pytest.approx(1.6, abs=TOL)
    assert bundle.a == pytest.approx(1.6, abs=TOL)
    # Random spot checks
    assert bundle.U == pytest.approx(0.8, abs=TOL)
    assert bundle.erlang_c == pytest.approx(2 * 0.8 ** 2 / 1.8, abs=TOL)
    # Pi sums to 1
    assert bundle.pi.sum() == pytest.approx(1.0, abs=TOL)
    # P(W_q > 2) coherent with formula
    expected = bundle.erlang_c * math.exp(-2.0 * MU * (1.0 - 0.8) * 2.0)
    assert bundle.P_wait_2 == pytest.approx(expected, abs=TOL)


def test_pi_vector_shape():
    n_max = 50
    pi = pi_vector_mm2(0.7, n_max=n_max)
    assert pi.shape == (n_max + 1,)


def test_pi_summation_to_unity_high_truncation():
    # Use truncation_n_max to ensure tail residual is below 1e-12 for every rho.
    for rho in RHOS:
        n_max = truncation_n_max(rho, tol=1e-14)
        pi = pi_vector_mm2(rho, n_max=n_max)
        assert abs(pi.sum() - 1.0) < 1e-12, f"rho={rho} sum error {pi.sum() - 1}"


def test_truncation_n_max_monotonic():
    # As tolerance tightens, n_max grows
    n_loose = truncation_n_max(0.9, tol=1e-6)
    n_tight = truncation_n_max(0.9, tol=1e-12)
    assert n_tight > n_loose
    # As rho approaches 1, n_max grows
    n_low = truncation_n_max(0.5, tol=1e-12)
    n_high = truncation_n_max(0.95, tol=1e-12)
    assert n_high > n_low
