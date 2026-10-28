"""Discrete-event simulator for the M/M/2 checkout queue."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# Result containers

@dataclass
class SimReplicationResult:
    rho: float
    mu: float
    lam: float
    seed: int
    n_customers: int
    warmup_customers: int
    sim_clock_end: float
    measurement_window: float

    pi_hat: np.ndarray

    L_hat: float
    L_q_hat: float
    U_hat: float

    W_hat: float
    W_q_hat: float

    erlang_c_hat: float
    fraction_no_wait_hat: float

    P_wait_t_hat: dict[float, float]
    P_Nq_ge_K_hat: dict[int, float]
    P_N_ge_Nstar_hat: dict[int, float]

    arrivals: int
    departures: int
    arrivals_post_warmup: int
    departures_post_warmup: int

    invariant_check: dict[str, float] = field(default_factory=dict)
    waits_in_queue: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class SimAggregateResult:
    rho: float
    mu: float
    lam: float
    n_replications: int
    pi_hat_mean: np.ndarray
    pi_hat_se: np.ndarray
    metric_means: dict[str, float]
    metric_sems: dict[str, float]
    metric_ci_low: dict[str, float]
    metric_ci_high: dict[str, float]
    raw_replications: list[SimReplicationResult] = field(default_factory=list)


# Single replication

_ARRIVAL = 0
_DEPARTURE = 1


def simulate_mm2_once(
    *,
    lam: float,
    mu: float,
    n_customers: int,
    warmup_fraction: float,
    seed: int,
    n_max_state: int = 60,
    wait_thresholds: Sequence[float] = (2.0, 3.0, 5.0),
    queue_thresholds: Sequence[int] = (3, 5, 10),
    stress_thresholds: Sequence[int] = (3, 5, 7, 10),
    record_waits: bool = False,
) -> SimReplicationResult:
    """Run a single discrete-event replication of the M/M/2 queue."""
    if lam <= 0 or mu <= 0:
        raise ValueError("lambda and mu must be positive.")
    if n_customers < 100:
        raise ValueError("n_customers must be at least 100.")
    if not (0.0 <= warmup_fraction < 1.0):
        raise ValueError("warmup_fraction must be in [0, 1).")
    rho = lam / (2.0 * mu)
    if rho >= 1.0:
        raise ValueError(f"rho = {rho} >= 1; the system is unstable.")

    rng = np.random.default_rng(seed)
    warmup = int(math.floor(warmup_fraction * n_customers))

    # Future-event list: (time, type_code, seq, payload). seq breaks ties.
    events: list[tuple[float, int, int, int | float]] = []
    seq_counter = 0

    def push(time: float, etype: int, payload: int | float) -> None:
        nonlocal seq_counter
        heapq.heappush(events, (time, etype, seq_counter, payload))
        seq_counter += 1

    # Simulation state
    clock = 0.0
    n_in_system = 0
    cashier_busy = [False, False]
    cashier_arrival_time: list[float] = [0.0, 0.0]
    cashier_entry_time: list[float] = [0.0, 0.0]
    queue_arrival_times: list[float] = []

    # Counters
    total_arrived = 0
    total_departed = 0
    arrivals_post_warmup = 0
    departures_post_warmup = 0
    erlang_c_hits = 0
    no_wait_hits = 0

    # Customer-average accumulators
    sum_W = 0.0
    sum_W_q = 0.0
    wait_threshold_counts = {float(t): 0 for t in wait_thresholds}
    waits_buffer: list[float] = []

    # Time-weighted accumulators
    state_time = np.zeros(n_max_state + 1, dtype=float)
    integral_N = 0.0
    integral_Nq = 0.0
    integral_busy = 0.0
    queue_threshold_time = {int(K): 0.0 for K in queue_thresholds}
    stress_threshold_time = {int(N): 0.0 for N in stress_thresholds}

    measurement_started = False
    measurement_start_time = 0.0
    last_event_time = 0.0

    def accumulate(t_new: float) -> None:
        nonlocal last_event_time, integral_N, integral_Nq, integral_busy
        if not measurement_started:
            return
        dt = t_new - last_event_time
        if dt < 0:
            if dt < -1e-9:
                raise RuntimeError(f"negative dt = {dt}")
            dt = 0.0
        if dt == 0.0:
            last_event_time = t_new
            return
        bucket = n_in_system if n_in_system <= n_max_state else n_max_state
        state_time[bucket] += dt
        integral_N += n_in_system * dt
        integral_Nq += max(n_in_system - 2, 0) * dt
        integral_busy += sum(cashier_busy) * dt
        for K in queue_threshold_time:
            if max(n_in_system - 2, 0) >= K:
                queue_threshold_time[K] += dt
        for Ns in stress_threshold_time:
            if n_in_system >= Ns:
                stress_threshold_time[Ns] += dt
        last_event_time = t_new

    # Bootstrap: schedule the first arrival
    push(rng.exponential(1.0 / lam), _ARRIVAL, 0)

    # Main event loop
    while total_departed < n_customers:
        if not events:
            raise RuntimeError("Event heap exhausted prematurely.")
        time, etype, _, payload = heapq.heappop(events)

        accumulate(time)
        clock = time

        if etype == _ARRIVAL:
            total_arrived += 1
            arrival_time = clock

            if measurement_started:
                arrivals_post_warmup += 1
                if n_in_system >= 2:
                    erlang_c_hits += 1
                else:
                    no_wait_hits += 1

            # Place into service or queue
            if not cashier_busy[0]:
                cashier_busy[0] = True
                cashier_arrival_time[0] = arrival_time
                cashier_entry_time[0] = arrival_time
                push(clock + rng.exponential(1.0 / mu), _DEPARTURE, 0)
                n_in_system += 1
            elif not cashier_busy[1]:
                cashier_busy[1] = True
                cashier_arrival_time[1] = arrival_time
                cashier_entry_time[1] = arrival_time
                push(clock + rng.exponential(1.0 / mu), _DEPARTURE, 1)
                n_in_system += 1
            else:
                queue_arrival_times.append(arrival_time)
                n_in_system += 1

            # Schedule next arrival
            push(clock + rng.exponential(1.0 / lam), _ARRIVAL, total_arrived)

            # Begin measurement at the (warmup + 1)-th arrival
            if (not measurement_started) and total_arrived == warmup + 1:
                measurement_started = True
                measurement_start_time = clock
                last_event_time = clock

        else:
            cashier = int(payload)
            if not cashier_busy[cashier]:
                raise RuntimeError(f"Departure for idle cashier {cashier}.")
            arr_t = cashier_arrival_time[cashier]
            entry_t = cashier_entry_time[cashier]
            sojourn = clock - arr_t
            wait_q = entry_t - arr_t
            n_in_system -= 1
            total_departed += 1

            # If queue is non-empty, put the next waiter into service
            if queue_arrival_times:
                next_arr = queue_arrival_times.pop(0)
                cashier_arrival_time[cashier] = next_arr
                cashier_entry_time[cashier] = clock
                push(clock + rng.exponential(1.0 / mu), _DEPARTURE, cashier)
            else:
                cashier_busy[cashier] = False
                cashier_arrival_time[cashier] = 0.0
                cashier_entry_time[cashier] = 0.0

            # Customer-average accumulation only for post-warmup arrivals
            if measurement_started and arr_t >= measurement_start_time:
                departures_post_warmup += 1
                sum_W += sojourn
                sum_W_q += wait_q
                if record_waits:
                    waits_buffer.append(wait_q)
                for t_thr in wait_threshold_counts:
                    if wait_q > t_thr:
                        wait_threshold_counts[t_thr] += 1

    # End-of-run accumulation
    accumulate(clock)
    measurement_window = clock - measurement_start_time
    if measurement_window <= 0.0:
        raise RuntimeError(
            "Measurement window is non-positive; increase n_customers or reduce warmup."
        )

    # Time-weighted estimators
    pi_hat = state_time / measurement_window
    L_hat = integral_N / measurement_window
    L_q_hat = integral_Nq / measurement_window
    U_hat = integral_busy / (2.0 * measurement_window)
    P_Nq_ge_K_hat = {K: queue_threshold_time[K] / measurement_window for K in queue_threshold_time}
    P_N_ge_Nstar_hat = {Ns: stress_threshold_time[Ns] / measurement_window for Ns in stress_threshold_time}

    # Customer-average estimators
    if departures_post_warmup == 0:
        raise RuntimeError("No post-warmup departures observed.")
    W_hat = sum_W / departures_post_warmup
    W_q_hat = sum_W_q / departures_post_warmup
    P_wait_t_hat = {
        t_thr: wait_threshold_counts[t_thr] / departures_post_warmup
        for t_thr in wait_threshold_counts
    }

    # PASTA estimators (arrival-fraction)
    if arrivals_post_warmup == 0:
        raise RuntimeError("No post-warmup arrivals observed.")
    erlang_c_hat = erlang_c_hits / arrivals_post_warmup
    fraction_no_wait_hat = no_wait_hits / arrivals_post_warmup

    # Invariant residuals (returned for diagnostic logging)
    invariant = {
        "L_minus_Lq_minus_2rho": float(L_hat - L_q_hat - 2.0 * rho),
        "littles_law_system_residual": float(L_hat / lam - W_hat),
        "littles_law_queue_residual": float(L_q_hat / lam - W_q_hat),
        "sojourn_decomposition_residual": float(W_hat - W_q_hat - 1.0 / mu),
    }

    return SimReplicationResult(
        rho=rho,
        mu=mu,
        lam=lam,
        seed=seed,
        n_customers=n_customers,
        warmup_customers=warmup,
        sim_clock_end=clock,
        measurement_window=measurement_window,
        pi_hat=pi_hat,
        L_hat=L_hat,
        L_q_hat=L_q_hat,
        U_hat=U_hat,
        W_hat=W_hat,
        W_q_hat=W_q_hat,
        erlang_c_hat=erlang_c_hat,
        fraction_no_wait_hat=fraction_no_wait_hat,
        P_wait_t_hat=P_wait_t_hat,
        P_Nq_ge_K_hat=P_Nq_ge_K_hat,
        P_N_ge_Nstar_hat=P_N_ge_Nstar_hat,
        arrivals=total_arrived,
        departures=total_departed,
        arrivals_post_warmup=arrivals_post_warmup,
        departures_post_warmup=departures_post_warmup,
        invariant_check=invariant,
        waits_in_queue=np.array(waits_buffer, dtype=float) if record_waits else np.empty(0),
    )


# Multiple replications with Student-t CI

def simulate_mm2_replications(
    *,
    lam: float,
    mu: float,
    n_customers: int,
    warmup_fraction: float,
    n_replications: int,
    base_seed: int,
    rho_index: int = 0,
    n_max_state: int = 60,
    wait_thresholds: Sequence[float] = (2.0, 3.0, 5.0),
    queue_thresholds: Sequence[int] = (3, 5, 10),
    stress_thresholds: Sequence[int] = (3, 5, 7, 10),
    record_waits_for_first: bool = False,
    confidence: float = 0.95,
) -> SimAggregateResult:
    """Run n_replications independent replications and return aggregated estimates with CIs."""
    if n_replications < 2:
        raise ValueError("n_replications must be at least 2 for CI estimation.")
    rho = lam / (2.0 * mu)

    reps: list[SimReplicationResult] = []
    for rep_index in range(n_replications):
        seed = base_seed + 1000 * rho_index + rep_index
        record = record_waits_for_first and rep_index == 0
        res = simulate_mm2_once(
            lam=lam,
            mu=mu,
            n_customers=n_customers,
            warmup_fraction=warmup_fraction,
            seed=seed,
            n_max_state=n_max_state,
            wait_thresholds=wait_thresholds,
            queue_thresholds=queue_thresholds,
            stress_thresholds=stress_thresholds,
            record_waits=record,
        )
        reps.append(res)

    # Aggregate scalar metrics
    metric_keys = [
        "L_hat", "L_q_hat", "U_hat", "W_hat", "W_q_hat",
        "erlang_c_hat", "fraction_no_wait_hat",
    ]
    arr_metrics = {k: np.array([getattr(r, k) for r in reps]) for k in metric_keys}
    for t in wait_thresholds:
        arr_metrics[f"P_wait_gt_{t:g}"] = np.array([r.P_wait_t_hat[float(t)] for r in reps])
    for K in queue_thresholds:
        arr_metrics[f"P_Nq_ge_{K}"] = np.array([r.P_Nq_ge_K_hat[int(K)] for r in reps])
    for Ns in stress_thresholds:
        arr_metrics[f"P_N_ge_{Ns}"] = np.array([r.P_N_ge_Nstar_hat[int(Ns)] for r in reps])

    means = {k: float(v.mean()) for k, v in arr_metrics.items()}
    sems = {k: float(v.std(ddof=1) / math.sqrt(n_replications)) for k, v in arr_metrics.items()}

    # Student-t CI
    from scipy import stats
    t_crit = stats.t.ppf(0.5 + confidence / 2.0, df=n_replications - 1)
    ci_low = {k: means[k] - t_crit * sems[k] for k in means}
    ci_high = {k: means[k] + t_crit * sems[k] for k in means}

    # Aggregate pi_hat across replications
    pi_stack = np.vstack([r.pi_hat for r in reps])
    pi_mean = pi_stack.mean(axis=0)
    pi_se = pi_stack.std(axis=0, ddof=1) / math.sqrt(n_replications)

    return SimAggregateResult(
        rho=rho,
        mu=mu,
        lam=lam,
        n_replications=n_replications,
        pi_hat_mean=pi_mean,
        pi_hat_se=pi_se,
        metric_means=means,
        metric_sems=sems,
        metric_ci_low=ci_low,
        metric_ci_high=ci_high,
        raw_replications=reps,
    )


__all__ = [
    "SimReplicationResult",
    "SimAggregateResult",
    "simulate_mm2_once",
    "simulate_mm2_replications",
]
