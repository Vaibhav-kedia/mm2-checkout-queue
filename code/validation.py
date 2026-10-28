"""Validation sweep: simulate, compare to theory, write CSV results."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from mmc_analytical import (
    L_mm2,
    L_q_mm2,
    P_Nq_ge_K_mm2,
    P_N_ge_Nstar_mm2,
    P_wait_gt_t_mm2,
    W_mm2,
    W_q_mm2,
    erlang_c_general,
    erlang_c_mm2,
    pi_n_mm2,
    utilization_mm2,
    W_q_general,
    P_wait_gt_t_general,
)
from mmc_simulator import (
    SimAggregateResult,
    simulate_mm2_replications,
)


# Configuration

RHOS = (0.30, 0.50, 0.70, 0.80, 0.90, 0.95)
MU = 1.0
R = 20
N = 200_000
WARMUP = 0.10
BASE_SEED = 20260422
WAIT_THRESHOLDS = (2.0, 3.0, 5.0)
QUEUE_THRESHOLDS = (3, 5, 10)
STRESS_THRESHOLDS = (3, 5, 7, 10)
N_MAX_STATE = 60
MC_UNRELIABLE_FLOOR = 1e-3

CONFIDENCE = 0.95
T_CRIT_R20 = float(stats.t.ppf(0.5 + CONFIDENCE / 2.0, df=R - 1))


# Result rows

@dataclass(frozen=True)
class MetricRow:
    rho: float
    metric: str
    theory: float
    sim_mean: float
    sim_se: float
    ci_low: float
    ci_high: float
    rel_err: float
    in_ci: bool
    mc_unreliable: bool


@dataclass(frozen=True)
class PiRow:
    rho: float
    n: int
    theory: float
    sim_mean: float
    sim_se: float
    ci_low: float
    ci_high: float
    in_ci: bool


# Sweep driver

def run_sweep(progress: bool = True) -> dict[float, SimAggregateResult]:
    """Run R replications at each rho and return a dict of aggregate results."""
    aggregates: dict[float, SimAggregateResult] = {}
    for idx, rho in enumerate(RHOS):
        lam = 2.0 * rho * MU
        if progress:
            print(f"  [validation] sweeping rho = {rho:.2f}, lam = {lam:.3f} ...")
        record_first = abs(rho - 0.80) < 1e-9
        agg = simulate_mm2_replications(
            lam=lam, mu=MU,
            n_customers=N,
            warmup_fraction=WARMUP,
            n_replications=R,
            base_seed=BASE_SEED,
            rho_index=idx,
            wait_thresholds=WAIT_THRESHOLDS,
            queue_thresholds=QUEUE_THRESHOLDS,
            stress_thresholds=STRESS_THRESHOLDS,
            n_max_state=N_MAX_STATE,
            record_waits_for_first=record_first,
            confidence=CONFIDENCE,
        )
        aggregates[rho] = agg
    return aggregates


# Theory lookup

def _theory_value(metric: str, rho: float) -> float:
    if metric == "L":
        return L_mm2(rho)
    if metric == "L_q":
        return L_q_mm2(rho)
    if metric == "W":
        return W_mm2(rho, MU)
    if metric == "W_q":
        return W_q_mm2(rho, MU)
    if metric == "U":
        return utilization_mm2(rho)
    if metric == "erlang_c":
        return erlang_c_mm2(rho)
    if metric == "fraction_no_wait":
        return 1.0 - erlang_c_mm2(rho)
    if metric.startswith("P_wait_gt_"):
        t = float(metric[len("P_wait_gt_"):])
        return P_wait_gt_t_mm2(rho, MU, t)
    if metric.startswith("P_Nq_ge_"):
        K = int(metric[len("P_Nq_ge_"):])
        return P_Nq_ge_K_mm2(rho, K)
    if metric.startswith("P_N_ge_"):
        Ns = int(metric[len("P_N_ge_"):])
        return P_N_ge_Nstar_mm2(rho, Ns)
    raise KeyError(f"Unknown metric label: {metric}")


_METRIC_ORDER = (
    ["L", "L_q", "W", "W_q", "U", "erlang_c", "fraction_no_wait"]
    + [f"P_wait_gt_{t:g}" for t in WAIT_THRESHOLDS]
    + [f"P_Nq_ge_{K}" for K in QUEUE_THRESHOLDS]
    + [f"P_N_ge_{Ns}" for Ns in STRESS_THRESHOLDS]
)

_METRIC_TO_SIM_KEY = {
    "L": "L_hat",
    "L_q": "L_q_hat",
    "W": "W_hat",
    "W_q": "W_q_hat",
    "U": "U_hat",
    "erlang_c": "erlang_c_hat",
    "fraction_no_wait": "fraction_no_wait_hat",
}
for t in WAIT_THRESHOLDS:
    _METRIC_TO_SIM_KEY[f"P_wait_gt_{t:g}"] = f"P_wait_gt_{t:g}"
for K in QUEUE_THRESHOLDS:
    _METRIC_TO_SIM_KEY[f"P_Nq_ge_{K}"] = f"P_Nq_ge_{K}"
for Ns in STRESS_THRESHOLDS:
    _METRIC_TO_SIM_KEY[f"P_N_ge_{Ns}"] = f"P_N_ge_{Ns}"


# Comparison rows

def collect_metric_rows(aggregates: dict[float, SimAggregateResult]) -> list[MetricRow]:
    """Build (rho, metric) comparison rows from sweep aggregates."""
    rows: list[MetricRow] = []
    for rho in RHOS:
        agg = aggregates[rho]
        for metric in _METRIC_ORDER:
            theory = _theory_value(metric, rho)
            sim_key = _METRIC_TO_SIM_KEY[metric]
            sim_mean = agg.metric_means[sim_key]
            sim_se = agg.metric_sems[sim_key]
            ci_low = agg.metric_ci_low[sim_key]
            ci_high = agg.metric_ci_high[sim_key]
            denom = max(abs(theory), 1e-12)
            rel_err = (sim_mean - theory) / denom
            in_ci = (ci_low - 1e-15) <= theory <= (ci_high + 1e-15)
            mc_unreliable = abs(theory) < MC_UNRELIABLE_FLOOR
            rows.append(MetricRow(
                rho=rho, metric=metric, theory=theory,
                sim_mean=sim_mean, sim_se=sim_se,
                ci_low=ci_low, ci_high=ci_high,
                rel_err=rel_err, in_ci=in_ci, mc_unreliable=mc_unreliable,
            ))
    return rows


def collect_pi_rows(
    aggregates: dict[float, SimAggregateResult],
    n_max_csv: int = 15,
) -> list[PiRow]:
    """Build (rho, n) stationary distribution comparison rows."""
    rows: list[PiRow] = []
    for rho in RHOS:
        agg = aggregates[rho]
        n_cap = min(n_max_csv, agg.pi_hat_mean.size - 1)
        for n in range(n_cap + 1):
            theory = pi_n_mm2(rho, n)
            sim_mean = float(agg.pi_hat_mean[n])
            sim_se = float(agg.pi_hat_se[n])
            ci_low = sim_mean - T_CRIT_R20 * sim_se
            ci_high = sim_mean + T_CRIT_R20 * sim_se
            in_ci = ci_low - 1e-15 <= theory <= ci_high + 1e-15
            rows.append(PiRow(
                rho=rho, n=n, theory=theory,
                sim_mean=sim_mean, sim_se=sim_se,
                ci_low=ci_low, ci_high=ci_high, in_ci=in_ci,
            ))
    return rows


# CSV writers

def write_metrics_csv(path: Path, rows: list[MetricRow]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rho", "metric", "theory", "sim_mean", "sim_se",
            "ci_low", "ci_high", "rel_err", "in_ci", "mc_unreliable",
        ])
        for r in rows:
            writer.writerow([
                f"{r.rho:.4f}", r.metric,
                f"{r.theory:.10g}", f"{r.sim_mean:.10g}", f"{r.sim_se:.10g}",
                f"{r.ci_low:.10g}", f"{r.ci_high:.10g}",
                f"{r.rel_err:.6g}", int(r.in_ci), int(r.mc_unreliable),
            ])


def write_pi_csv(path: Path, rows: list[PiRow]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rho", "n", "theory", "sim_mean", "sim_se",
            "ci_low", "ci_high", "in_ci",
        ])
        for r in rows:
            writer.writerow([
                f"{r.rho:.4f}", r.n,
                f"{r.theory:.10g}", f"{r.sim_mean:.10g}", f"{r.sim_se:.10g}",
                f"{r.ci_low:.10g}", f"{r.ci_high:.10g}",
                int(r.in_ci),
            ])


def write_summary_csv(path: Path, rows: list[MetricRow], pi_rows: list[PiRow]) -> None:
    rows_reliable = [r for r in rows if not r.mc_unreliable]
    pass_metric = sum(1 for r in rows_reliable if r.in_ci)
    total_metric = len(rows_reliable)
    pass_pi = sum(1 for r in pi_rows if r.in_ci)
    total_pi = len(pi_rows)
    overall_pass = pass_metric + pass_pi
    overall_total = total_metric + total_pi
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["category", "passes", "total", "pass_rate"])
        writer.writerow([
            "metrics_reliable",
            pass_metric, total_metric,
            f"{pass_metric / max(total_metric, 1):.4f}",
        ])
        writer.writerow([
            "pi_n",
            pass_pi, total_pi,
            f"{pass_pi / max(total_pi, 1):.4f}",
        ])
        writer.writerow([
            "overall",
            overall_pass, overall_total,
            f"{overall_pass / max(overall_total, 1):.4f}",
        ])


def write_waits_csv(path: Path, aggregates: dict[float, SimAggregateResult]) -> None:
    """Write per-customer queue waits at rho = 0.80 (first replication)."""
    rho_target = 0.80
    if rho_target not in aggregates:
        return
    agg = aggregates[rho_target]
    waits = agg.raw_replications[0].waits_in_queue
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wait_minutes"])
        for w in waits:
            writer.writerow([f"{w:.6g}"])


def write_mm3_comparison_csv(path: Path) -> None:
    """M/M/2 vs M/M/3 service-level curves on a fine lambda grid (mu = 1)."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "system", "c", "lam", "lam_per_hour", "rho",
            "erlang_c", "W_q", "P_W_q_gt_2",
        ])
        for lam in np.arange(0.05, 2.0, 0.025):
            rho = lam / 2.0
            if rho >= 0.999:
                continue
            C = erlang_c_mm2(rho)
            W_q = W_q_mm2(rho, MU)
            P2 = P_wait_gt_t_mm2(rho, MU, 2.0)
            writer.writerow([
                "MM2", 2, f"{lam:.4f}", f"{lam * 60:.2f}", f"{rho:.4f}",
                f"{C:.6g}", f"{W_q:.6g}", f"{P2:.6g}",
            ])
        for lam in np.arange(0.05, 3.0, 0.025):
            rho = lam / 3.0
            if rho >= 0.999:
                continue
            C = erlang_c_general(3, lam, MU)
            W_q = W_q_general(3, lam, MU)
            P2 = P_wait_gt_t_general(3, lam, MU, 2.0)
            writer.writerow([
                "MM3", 3, f"{lam:.4f}", f"{lam * 60:.2f}", f"{rho:.4f}",
                f"{C:.6g}", f"{W_q:.6g}", f"{P2:.6g}",
            ])


# Managerial threshold: max lambda such that P(W_q > t) <= target

def find_max_lambda_under_threshold(
    c: int, mu: float, t_threshold: float, target: float,
    lam_low: float = 0.001, lam_high: float | None = None, tol: float = 1e-5,
) -> float:
    """Bisection for the largest lam with P(W_q > t) <= target in M/M/c."""
    if lam_high is None:
        lam_high = c * mu - 1e-4
    if P_wait_gt_t_general(c, lam_low, mu, t_threshold) > target:
        return lam_low
    if P_wait_gt_t_general(c, lam_high, mu, t_threshold) <= target:
        return lam_high
    while lam_high - lam_low > tol:
        mid = 0.5 * (lam_low + lam_high)
        if P_wait_gt_t_general(c, mid, mu, t_threshold) <= target:
            lam_low = mid
        else:
            lam_high = mid
    return lam_low


def write_managerial_csv(path: Path) -> None:
    """Service-level capacities for c = 2 and c = 3 across (target, t)."""
    rows = []
    for target in (0.01, 0.05, 0.10):
        for t_thr in (2.0, 3.0, 5.0):
            for c in (2, 3):
                lam_max = find_max_lambda_under_threshold(c, MU, t_thr, target)
                rho_max = lam_max / (c * MU)
                rows.append({
                    "c": c,
                    "t_threshold_min": t_thr,
                    "target": target,
                    "lam_max_per_min": lam_max,
                    "lam_max_per_hour": lam_max * 60.0,
                    "rho_max": rho_max,
                })
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "c", "t_threshold_min", "target_P_W_q_gt_t",
            "lam_max_per_min", "lam_max_per_hour", "rho_max",
        ])
        for r in rows:
            writer.writerow([
                r["c"], f"{r['t_threshold_min']:.2f}", f"{r['target']:.3f}",
                f"{r['lam_max_per_min']:.6g}", f"{r['lam_max_per_hour']:.4g}",
                f"{r['rho_max']:.4f}",
            ])


# Public entry point

def run_validation(results_dir: Path) -> dict:
    """Run the full sweep and write all CSV outputs to results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    aggregates = run_sweep()

    metric_rows = collect_metric_rows(aggregates)
    pi_rows = collect_pi_rows(aggregates)

    write_metrics_csv(results_dir / "validation_metrics.csv", metric_rows)
    write_pi_csv(results_dir / "validation_pi.csv", pi_rows)
    write_summary_csv(results_dir / "validation_summary.csv", metric_rows, pi_rows)
    write_waits_csv(results_dir / "waits_at_rho_0p8.csv", aggregates)
    write_mm3_comparison_csv(results_dir / "mm3_comparison.csv")
    write_managerial_csv(results_dir / "managerial_thresholds.csv")

    # KS test for waiting-time distribution at rho = 0.80
    rho_t = 0.80
    waits = aggregates[rho_t].raw_replications[0].waits_in_queue
    waits_pos = waits[waits > 0]
    rate = 2.0 * MU * (1.0 - rho_t)
    ks_stat, ks_p = stats.kstest(waits_pos, "expon", args=(0.0, 1.0 / rate))

    return {
        "aggregates": aggregates,
        "metric_rows": metric_rows,
        "pi_rows": pi_rows,
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "ks_n": int(waits_pos.size),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", default="../results", type=Path,
        help="Directory in which CSVs will be written (default: ../results)",
    )
    args = parser.parse_args()
    output = run_validation(args.results_dir.resolve())
    print(f"Validation complete. KS at rho=0.80: stat={output['ks_stat']:.5f}, p={output['ks_p']:.5g}")
