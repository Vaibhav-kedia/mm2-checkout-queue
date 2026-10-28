"""Tests for mmc_simulator: input validation, reproducibility, invariant residuals, accuracy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmc_analytical import (
    L_mm2,
    L_q_mm2,
    W_mm2,
    W_q_mm2,
    erlang_c_mm2,
    P_wait_gt_t_mm2,
    P_Nq_ge_K_mm2,
    P_N_ge_Nstar_mm2,
)
from mmc_simulator import (
    simulate_mm2_once,
    simulate_mm2_replications,
)


# Input validation

def test_unstable_rho_raises():
    with pytest.raises(ValueError):
        simulate_mm2_once(
            lam=2.5, mu=1.0, n_customers=1000, warmup_fraction=0.1, seed=1,
        )


def test_negative_lambda_raises():
    with pytest.raises(ValueError):
        simulate_mm2_once(
            lam=-0.5, mu=1.0, n_customers=1000, warmup_fraction=0.1, seed=1,
        )


def test_n_customers_too_small_raises():
    with pytest.raises(ValueError):
        simulate_mm2_once(
            lam=0.5, mu=1.0, n_customers=10, warmup_fraction=0.1, seed=1,
        )


def test_warmup_out_of_range_raises():
    with pytest.raises(ValueError):
        simulate_mm2_once(
            lam=0.5, mu=1.0, n_customers=1000, warmup_fraction=1.0, seed=1,
        )
    with pytest.raises(ValueError):
        simulate_mm2_once(
            lam=0.5, mu=1.0, n_customers=1000, warmup_fraction=-0.1, seed=1,
        )


def test_too_few_replications_raises():
    with pytest.raises(ValueError):
        simulate_mm2_replications(
            lam=0.5, mu=1.0, n_customers=1000,
            warmup_fraction=0.1, n_replications=1, base_seed=1,
        )


# Bookkeeping correctness on a small run

def test_basic_book_keeping():
    res = simulate_mm2_once(
        lam=0.6, mu=1.0, n_customers=2000,
        warmup_fraction=0.1, seed=1, record_waits=True,
    )
    # Number processed
    assert res.departures == 2000
    assert res.arrivals >= res.departures  # there are still customers in flight
    # Pi_hat sums to 1
    assert abs(res.pi_hat.sum() - 1.0) < 1e-9
    # Probabilities are in [0, 1]
    assert (res.pi_hat >= 0).all()
    assert (res.pi_hat <= 1).all()
    # Estimators are non-negative and finite
    for v in (res.L_hat, res.L_q_hat, res.U_hat, res.W_hat, res.W_q_hat,
              res.erlang_c_hat, res.fraction_no_wait_hat):
        assert math.isfinite(v) and v >= 0
    # waits_in_queue is non-negative
    assert (res.waits_in_queue >= 0).all()


def test_no_wait_when_lam_very_small():
    # At very low load, virtually nobody waits
    res = simulate_mm2_once(
        lam=0.05, mu=1.0, n_customers=2000,
        warmup_fraction=0.1, seed=42,
    )
    assert res.W_q_hat < 0.05
    assert res.erlang_c_hat < 0.1
    assert res.fraction_no_wait_hat > 0.9


def test_seed_reproducibility():
    a = simulate_mm2_once(lam=0.6, mu=1.0, n_customers=1000,
                          warmup_fraction=0.1, seed=12345)
    b = simulate_mm2_once(lam=0.6, mu=1.0, n_customers=1000,
                          warmup_fraction=0.1, seed=12345)
    assert a.L_hat == b.L_hat
    assert a.W_q_hat == b.W_q_hat
    assert (a.pi_hat == b.pi_hat).all()


def test_different_seeds_differ():
    a = simulate_mm2_once(lam=0.6, mu=1.0, n_customers=1000,
                          warmup_fraction=0.1, seed=1)
    b = simulate_mm2_once(lam=0.6, mu=1.0, n_customers=1000,
                          warmup_fraction=0.1, seed=2)
    assert a.L_hat != b.L_hat


# Invariant checks at higher rho

def test_littles_law_holds_simulator():
    # At rho = 0.6 with 30k customers, Little's law residuals should be small.
    res = simulate_mm2_once(
        lam=1.2, mu=1.0, n_customers=30000,
        warmup_fraction=0.1, seed=7,
    )
    inv = res.invariant_check
    # |L/lam - W| should be small (under 5% of the analytical W)
    W_true = W_mm2(0.6, 1.0)
    assert abs(inv["littles_law_system_residual"]) < 0.05 * W_true
    assert abs(inv["littles_law_queue_residual"]) < 0.1 * max(W_true, 0.1)
    # Sojourn decomposition
    assert abs(inv["sojourn_decomposition_residual"]) < 0.05


def test_decomposition_L_minus_Lq_eq_2rho():
    # E[busy] = 2 rho. So L - L_q -> 2 rho with enough customers.
    res = simulate_mm2_once(
        lam=1.4, mu=1.0, n_customers=30000,
        warmup_fraction=0.1, seed=3,
    )
    rho = 0.7
    diff = res.L_hat - res.L_q_hat
    assert abs(diff - 2.0 * rho) < 0.05


# Accuracy test against closed-form theory

def test_accuracy_at_rho_0p7():
    # Run 5 replications of 50k customers at rho = 0.7. Each estimator
    # should be within 5% of its analytical value.
    rho = 0.7
    mu = 1.0
    lam = 2.0 * rho * mu
    agg = simulate_mm2_replications(
        lam=lam, mu=mu,
        n_customers=50000,
        warmup_fraction=0.1,
        n_replications=5,
        base_seed=20260422,
        rho_index=0,
    )
    # Compare aggregated means to theory
    L_th = L_mm2(rho)
    L_q_th = L_q_mm2(rho)
    W_th = W_mm2(rho, mu)
    W_q_th = W_q_mm2(rho, mu)
    C_th = erlang_c_mm2(rho)
    assert abs(agg.metric_means["L_hat"] - L_th) / L_th < 0.05
    assert abs(agg.metric_means["L_q_hat"] - L_q_th) / L_q_th < 0.10
    assert abs(agg.metric_means["W_hat"] - W_th) / W_th < 0.05
    assert abs(agg.metric_means["W_q_hat"] - W_q_th) / W_q_th < 0.10
    assert abs(agg.metric_means["erlang_c_hat"] - C_th) / C_th < 0.05
    assert abs(agg.metric_means["U_hat"] - rho) / rho < 0.05


def test_accuracy_pi_hat_at_rho_0p7():
    rho = 0.7
    mu = 1.0
    lam = 2.0 * rho * mu
    agg = simulate_mm2_replications(
        lam=lam, mu=mu,
        n_customers=50000,
        warmup_fraction=0.1,
        n_replications=5,
        base_seed=20260422,
        rho_index=0,
    )
    # Compare pi_hat[0..6] to theory
    from mmc_analytical import pi_n_mm2
    for n in range(7):
        sim = agg.pi_hat_mean[n]
        th = pi_n_mm2(rho, n)
        assert abs(sim - th) < 0.02, f"pi[{n}]: sim={sim}, theory={th}"


def test_replication_aggregation_outputs():
    agg = simulate_mm2_replications(
        lam=1.0, mu=1.0, n_customers=5000,
        warmup_fraction=0.1, n_replications=3, base_seed=42,
    )
    # Confidence intervals contain the means
    for k in agg.metric_means:
        assert agg.metric_ci_low[k] <= agg.metric_means[k] <= agg.metric_ci_high[k]
    # SEMs are non-negative
    for k in agg.metric_sems:
        assert agg.metric_sems[k] >= 0
    # pi means/se shapes match
    assert agg.pi_hat_mean.shape == agg.pi_hat_se.shape
    # Number of raw replications
    assert len(agg.raw_replications) == 3


def test_threshold_keys_match_input():
    res = simulate_mm2_once(
        lam=0.8, mu=1.0, n_customers=1000, warmup_fraction=0.1, seed=1,
        wait_thresholds=(1.0, 4.0),
        queue_thresholds=(2, 7),
        stress_thresholds=(2, 8),
    )
    assert set(res.P_wait_t_hat.keys()) == {1.0, 4.0}
    assert set(res.P_Nq_ge_K_hat.keys()) == {2, 7}
    assert set(res.P_N_ge_Nstar_hat.keys()) == {2, 8}


def test_record_waits_only_when_requested():
    on = simulate_mm2_once(lam=0.5, mu=1.0, n_customers=500,
                           warmup_fraction=0.1, seed=1, record_waits=True)
    off = simulate_mm2_once(lam=0.5, mu=1.0, n_customers=500,
                            warmup_fraction=0.1, seed=1, record_waits=False)
    assert on.waits_in_queue.size == on.departures_post_warmup
    assert off.waits_in_queue.size == 0
