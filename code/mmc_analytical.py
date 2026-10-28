"""Closed-form M/M/2 formulas (and general M/M/c routines for the M/M/3 staffing comparison)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


# Validation helpers

_MIN_POSITIVE = 1e-15


def _validate_rates(lam: float, mu: float) -> None:
    if not (math.isfinite(lam) and math.isfinite(mu)):
        raise ValueError("lambda and mu must be finite real numbers.")
    if lam <= 0:
        raise ValueError(f"lambda must be positive (got {lam}).")
    if mu <= 0:
        raise ValueError(f"mu must be positive (got {mu}).")


def _validate_rho(rho: float, *, strict_below_one: bool = True) -> None:
    if not math.isfinite(rho):
        raise ValueError("rho must be finite.")
    if rho <= 0:
        raise ValueError(f"rho must be strictly positive (got {rho}).")
    if strict_below_one and rho >= 1.0:
        raise ValueError(
            f"Stability requires rho < 1; got rho = {rho}. No stationary "
            "distribution exists."
        )


def rho_from_rates(lam: float, mu: float, c: int = 2) -> float:
    """Traffic intensity per server, rho = lam / (c * mu)."""
    _validate_rates(lam, mu)
    if c < 1:
        raise ValueError(f"Number of servers c must be >= 1 (got {c}).")
    return lam / (c * mu)


def offered_load(lam: float, mu: float) -> float:
    """Offered load in Erlangs: a = lam / mu."""
    _validate_rates(lam, mu)
    return lam / mu


# Stationary distribution of the M/M/2 chain

def pi0_mm2(rho: float) -> float:
    """pi_0 = (1 - rho) / (1 + rho), valid for rho < 1."""
    _validate_rho(rho)
    return (1.0 - rho) / (1.0 + rho)


def pi_n_mm2(rho: float, n: int) -> float:
    """Stationary probability of n customers; pi_n = 2 rho^n (1-rho)/(1+rho) for n >= 1."""
    _validate_rho(rho)
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"n must be a non-negative integer (got {n}).")
    if n == 0:
        return (1.0 - rho) / (1.0 + rho)
    return 2.0 * (rho ** n) * (1.0 - rho) / (1.0 + rho)


def truncation_n_max(rho: float, tol: float = 1e-13) -> int:
    """Smallest n_max such that the truncated tail sum_{n > n_max} pi_n is below tol."""
    _validate_rho(rho)
    if tol <= 0:
        raise ValueError(f"tol must be positive (got {tol}).")
    if rho < 1e-12:
        return 1
    needed = math.log(tol * (1.0 + rho) / 2.0) / math.log(rho)
    return max(1, int(math.ceil(needed)))


def pi_vector_mm2(rho: float, n_max: int = 200) -> np.ndarray:
    """Vector [pi_0, pi_1, ..., pi_{n_max}] from the closed form."""
    _validate_rho(rho)
    if n_max < 0:
        raise ValueError(f"n_max must be non-negative (got {n_max}).")
    pi = np.empty(n_max + 1, dtype=float)
    pi[0] = (1.0 - rho) / (1.0 + rho)
    if n_max >= 1:
        ns = np.arange(1, n_max + 1, dtype=float)
        pi[1:] = 2.0 * np.power(rho, ns) * (1.0 - rho) / (1.0 + rho)
    return pi


# Aggregate performance measures, M/M/2

def erlang_c_mm2(rho: float) -> float:
    """C(2, a) = P(N >= 2) = 2 rho^2 / (1 + rho)."""
    _validate_rho(rho)
    return 2.0 * rho * rho / (1.0 + rho)


def L_q_mm2(rho: float) -> float:
    """E[number waiting] = 2 rho^3 / (1 - rho^2)."""
    _validate_rho(rho)
    return 2.0 * rho ** 3 / (1.0 - rho * rho)


def L_mm2(rho: float) -> float:
    """E[number in system] = 2 rho / (1 - rho^2)."""
    _validate_rho(rho)
    return 2.0 * rho / (1.0 - rho * rho)


def W_q_mm2(rho: float, mu: float) -> float:
    """E[wait in queue] = rho^2 / (mu (1 - rho^2)). Little's law on the queue."""
    _validate_rho(rho)
    if mu <= 0:
        raise ValueError(f"mu must be positive (got {mu}).")
    return rho * rho / (mu * (1.0 - rho * rho))


def W_mm2(rho: float, mu: float) -> float:
    """E[time in system] = 1 / (mu (1 - rho^2))."""
    _validate_rho(rho)
    if mu <= 0:
        raise ValueError(f"mu must be positive (got {mu}).")
    return 1.0 / (mu * (1.0 - rho * rho))


def utilization_mm2(rho: float) -> float:
    """Per-cashier utilization U = rho."""
    _validate_rho(rho)
    return rho


def expected_busy_servers_mm2(rho: float) -> float:
    """E[busy cashiers] = 2 rho (offered load)."""
    _validate_rho(rho)
    return 2.0 * rho


# Distribution of the waiting time (M/M/2)

def P_wait_gt_t_mm2(rho: float, mu: float, t: float) -> float:
    """P(W_q > t) = C(2, a) * exp(-2 mu (1 - rho) t) for t >= 0."""
    _validate_rho(rho)
    if mu <= 0:
        raise ValueError(f"mu must be positive (got {mu}).")
    if t < 0:
        raise ValueError(f"t must be non-negative (got {t}).")
    return erlang_c_mm2(rho) * math.exp(-2.0 * mu * (1.0 - rho) * t)


def waiting_time_cdf_mm2(rho: float, mu: float, t: float) -> float:
    """CDF of W_q: P(W_q <= t) = 1 - C(2, a) * exp(-2 mu (1 - rho) t)."""
    _validate_rho(rho)
    if mu <= 0:
        raise ValueError(f"mu must be positive (got {mu}).")
    if t < 0:
        raise ValueError(f"t must be non-negative (got {t}).")
    return 1.0 - erlang_c_mm2(rho) * math.exp(-2.0 * mu * (1.0 - rho) * t)


def conditional_wait_rate_mm2(rho: float, mu: float) -> float:
    """Rate of W_q | W_q > 0 ~ Exp(2 mu (1 - rho))."""
    _validate_rho(rho)
    if mu <= 0:
        raise ValueError(f"mu must be positive (got {mu}).")
    return 2.0 * mu * (1.0 - rho)


# Operational checkpoint tail probabilities

def P_Nq_ge_K_mm2(rho: float, K: int) -> float:
    """P(N_q >= K). K = 0 returns the Erlang-C value; K >= 1 returns 2 rho^{K+2} / (1 + rho)."""
    _validate_rho(rho)
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError(f"K must be a non-negative integer (got {K}).")
    if K == 0:
        return erlang_c_mm2(rho)
    return 2.0 * (rho ** (K + 2)) / (1.0 + rho)


def P_N_ge_Nstar_mm2(rho: float, N_star: int) -> float:
    """Stress-state probability P(N >= N*)."""
    _validate_rho(rho)
    if not isinstance(N_star, (int, np.integer)) or N_star < 0:
        raise ValueError(f"N* must be a non-negative integer (got {N_star}).")
    if N_star == 0:
        return 1.0
    if N_star == 1:
        return 2.0 * rho / (1.0 + rho)
    return 2.0 * (rho ** N_star) / (1.0 + rho)


def P_no_wait_mm2(rho: float) -> float:
    """P(W_q = 0) = pi_0 + pi_1 = (1 - rho)(1 + 2 rho) / (1 + rho)."""
    _validate_rho(rho)
    return (1.0 - rho) * (1.0 + 2.0 * rho) / (1.0 + rho)


# General M/M/c (used only for the M/M/3 staffing comparison)

def erlang_c_general(c: int, lam: float, mu: float) -> float:
    """Erlang-C probability of waiting in an M/M/c queue."""
    _validate_rates(lam, mu)
    if c < 1:
        raise ValueError(f"c must be >= 1 (got {c}).")
    a = lam / mu
    rho = a / c
    _validate_rho(rho)

    term = 1.0
    sum_lower = 1.0
    for k in range(1, c):
        term *= a / k
        sum_lower += term
    term_c = term * a / c
    denom = (1.0 - rho) * sum_lower + term_c
    return term_c / denom


def W_q_general(c: int, lam: float, mu: float) -> float:
    """E[wait in queue] for M/M/c, via Erlang-C."""
    _validate_rates(lam, mu)
    if c < 1:
        raise ValueError(f"c must be >= 1 (got {c}).")
    a = lam / mu
    rho = a / c
    _validate_rho(rho)
    C = erlang_c_general(c, lam, mu)
    return C / (c * mu * (1.0 - rho))


def L_q_general(c: int, lam: float, mu: float) -> float:
    """E[number waiting] for M/M/c, via Little's law."""
    return lam * W_q_general(c, lam, mu)


def P_wait_gt_t_general(c: int, lam: float, mu: float, t: float) -> float:
    """P(W_q > t) for M/M/c."""
    if t < 0:
        raise ValueError(f"t must be non-negative (got {t}).")
    a = lam / mu
    rho = a / c
    _validate_rho(rho)
    return erlang_c_general(c, lam, mu) * math.exp(-c * mu * (1.0 - rho) * t)


# Bundle every metric for a (rho, mu) pair

@dataclass(frozen=True)
class MM2Theoretical:
    rho: float
    mu: float
    lam: float
    a: float
    pi0: float
    pi: np.ndarray
    erlang_c: float
    L: float
    L_q: float
    W: float
    W_q: float
    U: float
    P_wait_2: float
    P_wait_3: float
    P_wait_5: float
    P_Nq_ge_3: float
    P_Nq_ge_5: float
    P_Nq_ge_10: float
    P_N_ge_3: float
    P_N_ge_5: float
    P_N_ge_7: float
    P_N_ge_10: float


def all_metrics_mm2(rho: float, mu: float, n_max: int | None = None) -> MM2Theoretical:
    """Bundle every analytical quantity for a given (rho, mu) pair."""
    _validate_rho(rho)
    if mu <= 0:
        raise ValueError(f"mu must be positive (got {mu}).")
    if n_max is None:
        n_max = max(truncation_n_max(rho, tol=1e-14), 60)
    lam = 2.0 * rho * mu
    return MM2Theoretical(
        rho=rho,
        mu=mu,
        lam=lam,
        a=2.0 * rho,
        pi0=pi0_mm2(rho),
        pi=pi_vector_mm2(rho, n_max=n_max),
        erlang_c=erlang_c_mm2(rho),
        L=L_mm2(rho),
        L_q=L_q_mm2(rho),
        W=W_mm2(rho, mu),
        W_q=W_q_mm2(rho, mu),
        U=utilization_mm2(rho),
        P_wait_2=P_wait_gt_t_mm2(rho, mu, 2.0),
        P_wait_3=P_wait_gt_t_mm2(rho, mu, 3.0),
        P_wait_5=P_wait_gt_t_mm2(rho, mu, 5.0),
        P_Nq_ge_3=P_Nq_ge_K_mm2(rho, 3),
        P_Nq_ge_5=P_Nq_ge_K_mm2(rho, 5),
        P_Nq_ge_10=P_Nq_ge_K_mm2(rho, 10),
        P_N_ge_3=P_N_ge_Nstar_mm2(rho, 3),
        P_N_ge_5=P_N_ge_Nstar_mm2(rho, 5),
        P_N_ge_7=P_N_ge_Nstar_mm2(rho, 7),
        P_N_ge_10=P_N_ge_Nstar_mm2(rho, 10),
    )


def sweep_mm2(rhos: Iterable[float], mu: float = 1.0) -> list[MM2Theoretical]:
    """List of MM2Theoretical bundles, one per rho in rhos."""
    return [all_metrics_mm2(rho, mu) for rho in rhos]


__all__ = [
    "rho_from_rates",
    "offered_load",
    "truncation_n_max",
    "pi0_mm2",
    "pi_n_mm2",
    "pi_vector_mm2",
    "erlang_c_mm2",
    "L_q_mm2",
    "L_mm2",
    "W_q_mm2",
    "W_mm2",
    "utilization_mm2",
    "expected_busy_servers_mm2",
    "P_wait_gt_t_mm2",
    "waiting_time_cdf_mm2",
    "conditional_wait_rate_mm2",
    "P_Nq_ge_K_mm2",
    "P_N_ge_Nstar_mm2",
    "P_no_wait_mm2",
    "erlang_c_general",
    "W_q_general",
    "L_q_general",
    "P_wait_gt_t_general",
    "MM2Theoretical",
    "all_metrics_mm2",
    "sweep_mm2",
]
