"""Cross-module consistency tests: bundle agreement, KS test, PASTA verification."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from mmc_analytical import (
    L_mm2,
    L_q_mm2,
    P_Nq_ge_K_mm2,
    P_N_ge_Nstar_mm2,
    P_wait_gt_t_mm2,
    W_mm2,
    W_q_mm2,
    all_metrics_mm2,
    conditional_wait_rate_mm2,
    erlang_c_general,
    erlang_c_mm2,
    pi_n_mm2,
    pi_vector_mm2,
    truncation_n_max,
)
from mmc_simulator import (
    simulate_mm2_once,
    simulate_mm2_replications,
)


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
def test_bundle_internally_consistent(rho):
    bundle = all_metrics_mm2(rho, mu=1.0)
    # L = L_q + 2 rho
    assert bundle.L == pytest.approx(bundle.L_q + 2.0 * rho, abs=1e-12)
    # W - W_q = 1/mu
    assert bundle.W - bundle.W_q == pytest.approx(1.0, abs=1e-12)
    # Little's law on queue and on system
    assert bundle.W_q == pytest.approx(bundle.L_q / bundle.lam, abs=1e-12, rel=1e-12)
    assert bundle.W == pytest.approx(bundle.L / bundle.lam, abs=1e-12, rel=1e-12)
    # erlang_c == 2 rho^2 / (1 + rho)
    assert bundle.erlang_c == pytest.approx(2.0 * rho ** 2 / (1.0 + rho), abs=1e-13)
    # pi sums to 1
    assert bundle.pi.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("rho", [0.3, 0.7, 0.9])
def test_wait_tail_at_zero_equals_erlang_c(rho):
    assert P_wait_gt_t_mm2(rho, 1.0, 0.0) == pytest.approx(erlang_c_mm2(rho), abs=1e-13)


@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_general_mmc_specializes(rho):
    lam = 2.0 * rho * 1.0
    via_general = erlang_c_general(2, lam, 1.0)
    closed = erlang_c_mm2(rho)
    assert via_general == pytest.approx(closed, abs=1e-13)


def test_simulator_tracks_theory_at_rho_0p5():
    """At a friendly load with adequate horizon, simulator estimators should
    sit very close to theoretical values (within Monte Carlo noise)."""
    rho = 0.5
    mu = 1.0
    lam = 2.0 * rho * mu
    agg = simulate_mm2_replications(
        lam=lam, mu=mu,
        n_customers=80000,
        warmup_fraction=0.1,
        n_replications=5,
        base_seed=20260422,
        rho_index=10,
    )
    # Tight relative tolerances: 5% for headline metrics
    bundle = all_metrics_mm2(rho, mu)
    rel = lambda sim, th: abs(sim - th) / max(abs(th), 1e-9)
    assert rel(agg.metric_means["L_hat"], bundle.L) < 0.05
    assert rel(agg.metric_means["L_q_hat"], bundle.L_q) < 0.10
    assert rel(agg.metric_means["W_hat"], bundle.W) < 0.05
    assert rel(agg.metric_means["W_q_hat"], bundle.W_q) < 0.10
    assert rel(agg.metric_means["erlang_c_hat"], bundle.erlang_c) < 0.05
    assert rel(agg.metric_means["U_hat"], rho) < 0.05


def test_simulator_pi_hat_matches_at_rho_0p5():
    rho = 0.5
    mu = 1.0
    lam = 2.0 * rho * mu
    agg = simulate_mm2_replications(
        lam=lam, mu=mu,
        n_customers=80000,
        warmup_fraction=0.1,
        n_replications=5,
        base_seed=20260422,
        rho_index=11,
    )
    for n in range(8):
        th = pi_n_mm2(rho, n)
        assert abs(agg.pi_hat_mean[n] - th) < 0.01


def test_kolmogorov_smirnov_on_conditional_wait():
    """At rho = 0.8, conditional W_q | W_q > 0 should be Exp(2 mu (1 - rho))."""
    rho = 0.8
    mu = 1.0
    lam = 2.0 * rho * mu
    res = simulate_mm2_once(
        lam=lam, mu=mu, n_customers=100000, warmup_fraction=0.1,
        seed=20260422, record_waits=True,
    )
    waits_pos = res.waits_in_queue[res.waits_in_queue > 0]
    assert waits_pos.size > 5000  # enough for KS
    rate = conditional_wait_rate_mm2(rho, mu)  # = 2 * 1 * (1 - 0.8) = 0.4
    # Use SciPy's expon: scale = 1/rate
    stat, pval = stats.kstest(waits_pos, "expon", args=(0.0, 1.0 / rate))
    # Don't require a high p-value (KS is sensitive at large n), but the test
    # statistic itself should be small.
    assert stat < 0.05, f"KS stat = {stat}, p = {pval}"


def test_erlang_c_via_arrivals_matches_time_average():
    """PASTA: arrival-fraction seeing N >= 2 equals time-fraction P(N >= 2)."""
    rho = 0.7
    mu = 1.0
    lam = 2.0 * rho * mu
    agg = simulate_mm2_replications(
        lam=lam, mu=mu, n_customers=50000, warmup_fraction=0.1,
        n_replications=5, base_seed=999, rho_index=0,
    )
    arrival_C = agg.metric_means["erlang_c_hat"]
    # Time-average computed from pi_hat_mean
    time_C = agg.pi_hat_mean[2:].sum()
    assert abs(arrival_C - time_C) < 0.01


def test_l_q_decomposition_in_simulation():
    """E[busy] should equal L_hat - L_q_hat ≈ 2 rho."""
    rho = 0.6
    mu = 1.0
    lam = 2.0 * rho * mu
    agg = simulate_mm2_replications(
        lam=lam, mu=mu, n_customers=30000, warmup_fraction=0.1,
        n_replications=4, base_seed=111, rho_index=0,
    )
    diff = agg.metric_means["L_hat"] - agg.metric_means["L_q_hat"]
    assert abs(diff - 2.0 * rho) < 0.05


def test_truncation_helper_matches_actual_tail():
    """truncation_n_max(rho, tol) should produce a vector with tail < tol."""
    for rho in [0.3, 0.7, 0.95]:
        n_max = truncation_n_max(rho, tol=1e-10)
        pi = pi_vector_mm2(rho, n_max=n_max)
        tail = 1.0 - pi.sum()
        assert tail < 1e-10
