"""Builds the 11 figures from results CSVs and the analytical/simulator modules. Outputs PNG at 300 DPI."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap

import mmc_analytical as ma


# Style

_COLOR_THEORY = "#0B3D91"
_COLOR_SIM = "#C8102E"
_COLOR_ACCENT = "#1B998B"
_COLOR_GREY = "#444444"
_COLOR_LIGHT = "#BDBDBD"

_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "lines.linewidth": 1.8,
}

plt.rcParams.update(_RC)


# CSV reading helpers

@dataclass
class MetricRow:
    rho: float
    metric: str
    theory: float
    sim_mean: float
    sim_se: float
    ci_low: float
    ci_high: float
    rel_err: float
    in_ci: int
    mc_unreliable: int


def _read_metrics_csv(path: Path) -> list[MetricRow]:
    rows: list[MetricRow] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                MetricRow(
                    rho=float(r["rho"]),
                    metric=r["metric"],
                    theory=float(r["theory"]),
                    sim_mean=float(r["sim_mean"]),
                    sim_se=float(r["sim_se"]),
                    ci_low=float(r["ci_low"]),
                    ci_high=float(r["ci_high"]),
                    rel_err=float(r["rel_err"]),
                    in_ci=int(r["in_ci"]),
                    mc_unreliable=int(r["mc_unreliable"]),
                )
            )
    return rows


def _filter_metric(rows: list[MetricRow], metric: str) -> list[MetricRow]:
    return sorted([r for r in rows if r.metric == metric], key=lambda r: r.rho)


def _read_pi_csv(path: Path) -> dict[float, list[tuple[int, float, float, float, float, float]]]:
    """Return {rho: [(n, theory, sim_mean, sim_se, ci_low, ci_high), ...]}."""
    out: dict[float, list[tuple[int, float, float, float, float, float]]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out[float(r["rho"])].append(
                (
                    int(r["n"]),
                    float(r["theory"]),
                    float(r["sim_mean"]),
                    float(r["sim_se"]),
                    float(r["ci_low"]),
                    float(r["ci_high"]),
                )
            )
    for rho in out:
        out[rho].sort(key=lambda t: t[0])
    return out


def _read_waits(path: Path) -> np.ndarray:
    waits: list[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            waits.append(float(r["wait_minutes"]))
    return np.asarray(waits, dtype=float)


def _read_mm3(path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = defaultdict(list)
    sysname: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sysname.append(r["system"])
            for k in ("c", "lam", "lam_per_hour", "rho", "erlang_c", "W_q", "P_W_q_gt_2"):
                cols[k].append(float(r[k]))
    arr = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    arr["system"] = np.asarray(sysname)
    return arr


# Figure 1: rate diagram

def fig01_rate_diagram(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.6, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")

    states = [0, 1, 2, 3, 4]
    centers = [(2 * i, 0) for i in range(len(states))]
    radius = 0.45

    for (cx, cy), n in zip(centers, states):
        circ = Circle((cx, cy), radius, facecolor="white",
                      edgecolor=_COLOR_THEORY, linewidth=1.8, zorder=3)
        ax.add_patch(circ)
        ax.text(cx, cy, f"{n}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=_COLOR_THEORY, zorder=4)

    # ellipsis to indicate continuation
    ellipsis_x = centers[-1][0] + 1.3
    ax.text(ellipsis_x, 0, r"$\cdots$", ha="center", va="center", fontsize=18, color=_COLOR_GREY)

    # Birth arrows above (lambda)
    for i in range(len(states) - 1):
        x0 = centers[i][0] + radius
        x1 = centers[i + 1][0] - radius
        arr = FancyArrowPatch((x0, 0.18), (x1, 0.18), arrowstyle="->",
                              mutation_scale=14, color=_COLOR_THEORY, linewidth=1.5)
        ax.add_patch(arr)
        ax.text((x0 + x1) / 2, 0.55, r"$\lambda$", ha="center", va="center",
                fontsize=12, color=_COLOR_THEORY)

    # arrow into the ellipsis
    arr = FancyArrowPatch((centers[-1][0] + radius, 0.18),
                          (ellipsis_x - 0.25, 0.18), arrowstyle="->",
                          mutation_scale=14, color=_COLOR_THEORY, linewidth=1.5)
    ax.add_patch(arr)
    ax.text((centers[-1][0] + radius + ellipsis_x - 0.25) / 2, 0.55,
            r"$\lambda$", ha="center", va="center", fontsize=12, color=_COLOR_THEORY)

    # Death arrows below
    death_labels = [r"$\mu$", r"$2\mu$", r"$2\mu$", r"$2\mu$"]
    for i in range(len(states) - 1):
        x1 = centers[i + 1][0] - radius
        x0 = centers[i][0] + radius
        arr = FancyArrowPatch((x1, -0.18), (x0, -0.18), arrowstyle="->",
                              mutation_scale=14, color=_COLOR_SIM, linewidth=1.5)
        ax.add_patch(arr)
        ax.text((x0 + x1) / 2, -0.6, death_labels[i], ha="center", va="center",
                fontsize=12, color=_COLOR_SIM)

    # tail back-arrow
    arr = FancyArrowPatch((ellipsis_x - 0.25, -0.18),
                          (centers[-1][0] + radius, -0.18),
                          arrowstyle="->", mutation_scale=14,
                          color=_COLOR_SIM, linewidth=1.5)
    ax.add_patch(arr)
    ax.text((centers[-1][0] + radius + ellipsis_x - 0.25) / 2, -0.6, r"$2\mu$",
            ha="center", va="center", fontsize=12, color=_COLOR_SIM)

    ax.text(5, 1.45, "M/M/2 birth and death chain",
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(5, -1.25,
            r"Birth rate $\lambda$ from every state. Death rate $\mu$ from state 1; $2\mu$ from every state $n\geq 2$.",
            ha="center", va="center", fontsize=10, color=_COLOR_GREY)

    fig.savefig(out, dpi=300)
    plt.close(fig)


# Figure 2: warmup convergence

def fig02_warmup_convergence(out: Path) -> None:
    """Run a single sample path and show the running-average L converging to theory.

    Customers are accumulated and the running time average of N is plotted.
    """
    rho = 0.85
    mu = 1.0
    lam = 2.0 * rho * mu
    n_customers = 80_000

    # We will run our own sample path and accumulate L on the fly.
    rng = np.random.default_rng(20260427)

    # Inline DES with the same conventions as mmc_simulator, but we record the
    # running L as a function of completed customers.
    import heapq
    from itertools import count

    seq_counter = count()
    heap: list = []
    queue: list[float] = []
    busy = 0
    n_in_system = 0
    t = 0.0
    arrivals = 0
    departures = 0
    last_event_time = 0.0
    area_N = 0.0
    n_running: list[int] = []
    L_running: list[float] = []
    sample_times: list[float] = []

    # schedule first arrival
    heapq.heappush(heap, (rng.exponential(1.0 / lam), 0, next(seq_counter), None))

    while arrivals < n_customers and heap:
        t_event, code, _, _ = heapq.heappop(heap)
        area_N += n_in_system * (t_event - last_event_time)
        last_event_time = t_event
        t = t_event

        if code == 0:  # arrival
            arrivals += 1
            n_in_system += 1
            if busy < 2:
                busy += 1
                svc = rng.exponential(1.0 / mu)
                heapq.heappush(heap, (t + svc, 1, next(seq_counter), None))
            else:
                queue.append(t)
            if arrivals < n_customers:
                heapq.heappush(heap, (t + rng.exponential(1.0 / lam), 0, next(seq_counter), None))
        else:  # departure
            departures += 1
            n_in_system -= 1
            if queue:
                queue.pop(0)
                svc = rng.exponential(1.0 / mu)
                heapq.heappush(heap, (t + svc, 1, next(seq_counter), None))
            else:
                busy -= 1

            if departures % 200 == 0 and t > 0:
                sample_times.append(t)
                L_running.append(area_N / t)

    L_running_arr = np.asarray(L_running)
    times_arr = np.asarray(sample_times)
    L_theory = ma.L_mm2(rho)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(times_arr, L_running_arr, color=_COLOR_SIM, alpha=0.85,
            label=r"Running time-average of $N(t)$")
    ax.axhline(L_theory, color=_COLOR_THEORY, linestyle="--",
               label=fr"Theoretical $L = {L_theory:.4f}$")

    cutoff = times_arr[len(times_arr) // 10]
    ax.axvline(cutoff, color=_COLOR_GREY, linestyle=":", linewidth=1.2,
               label="Warmup cutoff (first 10% of customers)")

    ax.set_xlabel("Simulated time (minutes)")
    ax.set_ylabel(r"Running average of $N(t)$")
    ax.set_title(rf"Warmup convergence at $\rho = {rho}$ ($\lambda = {lam}$, $\mu = {mu}$)")
    ax.legend(loc="lower right")

    fig.savefig(out, dpi=300)
    plt.close(fig)


# Figure 3: stationary distribution

def fig03_stationary_distribution(pi_data: dict, out: Path) -> None:
    rhos_to_plot = [0.50, 0.80, 0.95]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=False)

    for ax, rho in zip(axes, rhos_to_plot):
        rows = pi_data[rho]
        n_arr = np.array([r[0] for r in rows])
        theory = np.array([r[1] for r in rows])
        sim = np.array([r[2] for r in rows])
        ci_low = np.array([r[4] for r in rows])
        ci_high = np.array([r[5] for r in rows])
        err_low = sim - ci_low
        err_high = ci_high - sim

        # show first 12 states for readability
        m = 12 if rho < 0.95 else 16
        n_arr = n_arr[:m]
        theory = theory[:m]
        sim = sim[:m]
        err_low = err_low[:m]
        err_high = err_high[:m]

        width = 0.4
        ax.bar(n_arr - width / 2, theory, width=width,
               color=_COLOR_THEORY, alpha=0.75, label="Theory")
        ax.bar(n_arr + width / 2, sim, width=width,
               color=_COLOR_SIM, alpha=0.75, label="Simulation",
               yerr=[err_low, err_high], capsize=2.5,
               error_kw={"elinewidth": 0.8, "ecolor": _COLOR_GREY})
        ax.set_xlabel(r"State $n$")
        ax.set_ylabel(r"$\pi_n$")
        ax.set_title(rf"$\rho = {rho:.2f}$")
        ax.set_xticks(np.arange(0, m, 2))
        if rho == rhos_to_plot[0]:
            ax.legend(loc="upper right")

    fig.suptitle(r"Stationary distribution $\pi_n$: theory vs simulation",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# Figure 4: core metrics vs rho

def fig04_core_metrics(metric_rows: list[MetricRow], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    rho_curve = np.linspace(0.05, 0.97, 400)

    panels = [
        ("L_q", axes[0, 0], r"$L_q$ (expected queue length)",
         lambda r: ma.L_q_mm2(r)),
        ("W_q", axes[0, 1], r"$W_q$ (expected waiting time, minutes)",
         lambda r: ma.W_q_mm2(r, 1.0)),
        ("L", axes[1, 0], r"$L$ (expected number in system)",
         lambda r: ma.L_mm2(r)),
        ("U", axes[1, 1], r"$U$ (per-cashier utilization)",
         lambda r: ma.utilization_mm2(r)),
    ]

    for metric, ax, title, theory_fn in panels:
        rows = _filter_metric(metric_rows, metric)
        rhos = np.array([r.rho for r in rows])
        sim = np.array([r.sim_mean for r in rows])
        lo = np.array([r.ci_low for r in rows])
        hi = np.array([r.ci_high for r in rows])
        err = np.vstack([sim - lo, hi - sim])
        theory = np.array([theory_fn(r) for r in rho_curve])

        ax.plot(rho_curve, theory, color=_COLOR_THEORY, label="Theory", zorder=2)
        ax.errorbar(rhos, sim, yerr=err, fmt="o",
                    color=_COLOR_SIM, ecolor=_COLOR_SIM, capsize=3,
                    markersize=5, label="Simulation (95% CI)", zorder=3)
        ax.set_xlabel(r"Traffic intensity $\rho$")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc="upper left")
        if metric in ("L_q", "W_q", "L"):
            ax.set_yscale("log")

    fig.suptitle("Core M/M/2 performance measures: theory vs simulation",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# Figure 5: Erlang C

def fig05_erlang_c(metric_rows: list[MetricRow], out: Path) -> None:
    rows = _filter_metric(metric_rows, "erlang_c")
    rhos = np.array([r.rho for r in rows])
    sim = np.array([r.sim_mean for r in rows])
    lo = np.array([r.ci_low for r in rows])
    hi = np.array([r.ci_high for r in rows])
    err = np.vstack([sim - lo, hi - sim])

    rho_curve = np.linspace(0.01, 0.97, 400)
    theory = 2 * rho_curve ** 2 / (1 + rho_curve)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(rho_curve, theory, color=_COLOR_THEORY,
            label=r"Theory: $C(2,a) = 2\rho^2/(1+\rho)$")
    ax.errorbar(rhos, sim, yerr=err, fmt="o",
                color=_COLOR_SIM, ecolor=_COLOR_SIM, capsize=3,
                markersize=6, label="Simulation (95% CI)")
    ax.set_xlabel(r"Traffic intensity $\rho$")
    ax.set_ylabel(r"$C(2, a) = P(\mathrm{wait} > 0)$")
    ax.set_title(r"Erlang C probability of waiting: theory vs simulation")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.02)

    fig.savefig(out, dpi=300)
    plt.close(fig)


# Figure 6: waiting-time CDF

def fig06_waiting_time_cdf(waits: np.ndarray, out: Path) -> None:
    rho = 0.80
    mu = 1.0
    C = ma.erlang_c_mm2(rho)
    rate = 2 * mu * (1 - rho)

    # full empirical CDF (includes zeros for those who did not wait)
    waits_sorted = np.sort(waits)
    n = waits_sorted.size
    ecdf = np.arange(1, n + 1) / n

    # sample to keep the plot file size reasonable
    if n > 5000:
        idx = np.linspace(0, n - 1, 5000).astype(int)
        x_e = waits_sorted[idx]
        y_e = ecdf[idx]
    else:
        x_e = waits_sorted
        y_e = ecdf

    t_grid = np.linspace(0, max(waits_sorted.max(), 8.0), 500)
    cdf_theory = 1 - C * np.exp(-rate * t_grid)
    cdf_theory[0] = max(cdf_theory[0], 1 - C)  # the actual jump at 0 is to 1-C

    # KS statistic on the conditional positive waits
    pos_waits = waits[waits > 0]
    pos_sorted = np.sort(pos_waits)
    n_pos = pos_sorted.size
    ecdf_pos = np.arange(1, n_pos + 1) / n_pos
    cdf_cond_theory_at = 1 - np.exp(-rate * pos_sorted)
    ks_stat = float(np.max(np.abs(ecdf_pos - cdf_cond_theory_at)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.plot(x_e, y_e, color=_COLOR_SIM, drawstyle="steps-post",
            label="Empirical CDF (simulation)")
    ax.plot(t_grid, cdf_theory, color=_COLOR_THEORY, linestyle="--",
            label=r"Theory: $1 - C\,e^{-2\mu(1-\rho)t}$")
    ax.scatter([0], [1 - C], color=_COLOR_THEORY, zorder=5)
    ax.text(0.15, 1 - C - 0.02, fr"$P(W_q = 0) = {1 - C:.3f}$",
            color=_COLOR_THEORY, fontsize=10)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Wait time $t$ (minutes)")
    ax.set_ylabel(r"$P(W_q \leq t)$")
    ax.set_title(rf"Unconditional waiting time CDF at $\rho = {rho:.2f}$")
    ax.legend(loc="lower right")

    ax = axes[1]
    cond_theory = 1 - np.exp(-rate * t_grid)
    ax.plot(pos_sorted[::max(1, n_pos // 5000)],
            ecdf_pos[::max(1, n_pos // 5000)],
            color=_COLOR_SIM, drawstyle="steps-post",
            label=r"Empirical $W_q \mid W_q > 0$")
    ax.plot(t_grid, cond_theory, color=_COLOR_THEORY, linestyle="--",
            label=r"Theory: $\mathrm{Exp}(2\mu(1-\rho))$")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Wait time $t$ (minutes)")
    ax.set_ylabel(r"$P(W_q \leq t \mid W_q > 0)$")
    ax.set_title(rf"Conditional waiting-time CDF (KS stat = {ks_stat:.4f})")
    ax.legend(loc="lower right")

    fig.suptitle(rf"Waiting-time validation at $\rho = {rho:.2f}$, $\mu = {mu}$",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# Figure 7: stress near unity

def fig07_stress_near_unity(out: Path) -> None:
    rho = np.linspace(0.05, 0.99, 500)
    L_q = 2 * rho ** 3 / (1 - rho ** 2)
    W_q = rho ** 2 / (1 - rho ** 2)  # mu=1
    asymp = rho / (1 - rho)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.semilogy(rho, L_q, color=_COLOR_THEORY, label=r"$L_q = 2\rho^3/(1-\rho^2)$")
    ax.semilogy(rho, asymp, color=_COLOR_GREY, linestyle="--",
                label=r"Asymptote: $\rho/(1-\rho)$")
    ax.set_xlabel(r"Traffic intensity $\rho$")
    ax.set_ylabel(r"$L_q$  (log scale)")
    ax.set_title(r"Queue length explodes as $\rho \to 1^-$")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.semilogy(rho, W_q, color=_COLOR_THEORY,
                label=r"$W_q = \rho^2/(\mu(1-\rho^2))$")
    ax.semilogy(rho, rho / (1 - rho), color=_COLOR_GREY, linestyle="--",
                label=r"Asymptote: $\rho/(\mu(1-\rho))$")
    ax.set_xlabel(r"Traffic intensity $\rho$")
    ax.set_ylabel(r"$W_q$ in minutes  (log scale)")
    ax.set_title(r"Waiting time explodes as $\rho \to 1^-$")
    ax.legend(loc="upper left")

    fig.suptitle(r"Stability: divergence of mean queue length and waiting time near $\rho=1$",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# Figure 8: pooling comparison: M/M/2 vs M/M/1 with rate 2 mu

def fig08_pooling_comparison(out: Path) -> None:
    """Compare two cashiers at rate mu (M/M/2) against one cashier at rate 2 mu (M/M/1).

    The single super-cashier matches total service capacity; the comparison
    isolates the cost of capacity fragmentation.
    """
    rho = np.linspace(0.05, 0.97, 400)
    mu = 1.0

    # M/M/2: lambda = 2*rho*mu, the standard expressions
    Lq_mm2 = 2 * rho ** 3 / (1 - rho ** 2)
    Wq_mm2 = rho ** 2 / (mu * (1 - rho ** 2))

    # M/M/1 with combined rate 2 mu and the same arrival rate => same rho:
    # Lq = rho^2 / (1-rho), Wq = Lq / lambda = rho / (2 mu (1-rho))
    Lq_mm1 = rho ** 2 / (1 - rho)
    Wq_mm1 = rho / (2 * mu * (1 - rho))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.plot(rho, Lq_mm2, color=_COLOR_THEORY, label="Two cashiers, rate $\\mu$ each (M/M/2)")
    ax.plot(rho, Lq_mm1, color=_COLOR_ACCENT, linestyle="--",
            label="One super-cashier, rate $2\\mu$ (M/M/1)")
    ax.set_yscale("log")
    ax.set_xlabel(r"Traffic intensity $\rho$")
    ax.set_ylabel(r"$L_q$  (log scale)")
    ax.set_title(r"Expected queue length")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(rho, Wq_mm2, color=_COLOR_THEORY, label="Two cashiers, rate $\\mu$ each (M/M/2)")
    ax.plot(rho, Wq_mm1, color=_COLOR_ACCENT, linestyle="--",
            label="One super-cashier, rate $2\\mu$ (M/M/1)")
    ax.set_yscale("log")
    ax.set_xlabel(r"Traffic intensity $\rho$")
    ax.set_ylabel(r"$W_q$ in minutes  (log scale)")
    ax.set_title(r"Expected waiting time")
    ax.legend(loc="upper left")

    fig.suptitle("Pooling comparison: two cashiers at rate $\\mu$ versus one cashier at rate $2\\mu$",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# Figure 9: validation scorecard

def fig09_validation_scorecard(metric_rows: list[MetricRow], out: Path) -> None:
    """Heatmap of |relative error| across (metric, rho) pairs, annotated with
    a check or cross indicating CI containment."""

    # Restrict to the operationally most relevant metrics (theory > 1e-3 region)
    headline = ["L", "L_q", "W", "W_q", "U", "erlang_c"]
    checkpoints = ["P_wait_gt_2", "P_wait_gt_3", "P_wait_gt_5",
                   "P_Nq_ge_3", "P_Nq_ge_5", "P_Nq_ge_10",
                   "P_N_ge_3", "P_N_ge_5", "P_N_ge_7", "P_N_ge_10",
                   "fraction_no_wait"]
    metric_order = headline + checkpoints

    rho_order = sorted({r.rho for r in metric_rows})

    # Build matrix of |rel_err| in percent
    matrix = np.full((len(metric_order), len(rho_order)), np.nan)
    flag_ci = np.zeros_like(matrix, dtype=int)
    flag_unrel = np.zeros_like(matrix, dtype=int)

    by_key = {(r.metric, r.rho): r for r in metric_rows}
    for i, m in enumerate(metric_order):
        for j, rho in enumerate(rho_order):
            r = by_key.get((m, rho))
            if r is None:
                continue
            matrix[i, j] = 100.0 * abs(r.rel_err)
            flag_ci[i, j] = r.in_ci
            flag_unrel[i, j] = r.mc_unreliable

    # Truncate display to keep the scale meaningful
    display = np.clip(matrix, 0, 5.0)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    cmap = LinearSegmentedColormap.from_list(
        "scorecard", ["#0B6B3A", "#E8E8C0", "#C8102E"], N=256)
    im = ax.imshow(display, cmap=cmap, aspect="auto", vmin=0, vmax=5.0)

    ax.set_xticks(np.arange(len(rho_order)))
    ax.set_xticklabels([f"{r:.2f}" for r in rho_order])
    ax.set_yticks(np.arange(len(metric_order)))
    ax.set_yticklabels(metric_order)
    ax.set_xlabel(r"Traffic intensity $\rho$")
    ax.set_title("Validation scorecard: |relative error| (%) across (metric, $\\rho$) pairs")

    # Annotate each cell
    for i in range(len(metric_order)):
        for j in range(len(rho_order)):
            if np.isnan(matrix[i, j]):
                continue
            v = matrix[i, j]
            label = f"{v:.2f}"
            if flag_unrel[i, j]:
                label = f"{label}*"
            color = "white" if v > 2.5 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
            # CI marker on the right edge
            mark = "✓" if flag_ci[i, j] else "✗"
            mark_color = "#0B6B3A" if flag_ci[i, j] else "#C8102E"
            ax.text(j + 0.36, i + 0.30, mark, ha="center", va="center",
                    fontsize=9, color=mark_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("|relative error| (%, capped at 5%)")

    n_pass = int(flag_ci.sum())
    n_total = int((~np.isnan(matrix)).sum())
    pct = 100.0 * n_pass / n_total
    ax.text(0.0, len(metric_order) - 0.0, "", transform=ax.transAxes)
    fig.text(0.5, -0.02,
             f"CI-containment pass rate: {n_pass}/{n_total} = {pct:.2f}%   "
             f"(asterisk = MC unreliable; checkmark = simulation CI contains theory)",
             ha="center", fontsize=10, color=_COLOR_GREY)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# Figure 10: checkpoint heatmap

def fig10_checkpoint_heatmap(out: Path) -> None:
    """Theoretical operational checkpoints across rho, on a uniform color scale."""

    rhos = np.array([0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
    mu = 1.0

    rows = []
    labels = []

    rows.append([ma.P_wait_gt_t_mm2(r, mu, 2.0) for r in rhos]); labels.append(r"$P(W_q > 2)$")
    rows.append([ma.P_wait_gt_t_mm2(r, mu, 3.0) for r in rhos]); labels.append(r"$P(W_q > 3)$")
    rows.append([ma.P_wait_gt_t_mm2(r, mu, 5.0) for r in rhos]); labels.append(r"$P(W_q > 5)$")
    rows.append([ma.P_Nq_ge_K_mm2(r, 3) for r in rhos]); labels.append(r"$P(N_q \geq 3)$")
    rows.append([ma.P_Nq_ge_K_mm2(r, 5) for r in rhos]); labels.append(r"$P(N_q \geq 5)$")
    rows.append([ma.P_Nq_ge_K_mm2(r, 10) for r in rhos]); labels.append(r"$P(N_q \geq 10)$")
    rows.append([ma.P_N_ge_Nstar_mm2(r, 3) for r in rhos]); labels.append(r"$P(N \geq 3)$")
    rows.append([ma.P_N_ge_Nstar_mm2(r, 5) for r in rhos]); labels.append(r"$P(N \geq 5)$")
    rows.append([ma.P_N_ge_Nstar_mm2(r, 7) for r in rhos]); labels.append(r"$P(N \geq 7)$")
    rows.append([ma.P_N_ge_Nstar_mm2(r, 10) for r in rhos]); labels.append(r"$P(N \geq 10)$")
    rows.append([ma.erlang_c_mm2(r) for r in rhos]); labels.append(r"$P(\mathrm{wait} > 0)$")

    M = np.array(rows)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    cmap = LinearSegmentedColormap.from_list(
        "checkpoint", ["#FFFFFF", "#FFC966", "#C8102E"], N=256)
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(rhos)))
    ax.set_xticklabels([f"{r:.2f}" for r in rhos])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Traffic intensity $\rho$")
    ax.set_title("Operational checkpoints: theoretical probabilities across loads")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v < 1e-4:
                txt = f"{v:.1e}"
            elif v < 1e-2:
                txt = f"{v:.4f}"
            else:
                txt = f"{v:.3f}"
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Probability")

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# Figure 11: M/M/2 vs M/M/3 staffing comparison

def fig11_staffing_comparison(mm3_data: dict, out: Path) -> None:
    sysarr = mm3_data["system"]
    mask2 = (sysarr == "MM2")
    mask3 = (sysarr == "MM3")

    lam2 = mm3_data["lam_per_hour"][mask2]
    lam3 = mm3_data["lam_per_hour"][mask3]
    Wq2 = mm3_data["W_q"][mask2]
    Wq3 = mm3_data["W_q"][mask3]
    Pw2 = mm3_data["P_W_q_gt_2"][mask2]
    Pw3 = mm3_data["P_W_q_gt_2"][mask3]
    C2 = mm3_data["erlang_c"][mask2]
    C3 = mm3_data["erlang_c"][mask3]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    ax = axes[0]
    ax.plot(lam2, Wq2, color=_COLOR_THEORY, label="M/M/2 (two cashiers)")
    ax.plot(lam3, Wq3, color=_COLOR_ACCENT, label="M/M/3 (three cashiers)")
    ax.axhline(2.0, color=_COLOR_GREY, linestyle=":",
               label=r"$W_q = 2$ minutes target")
    ax.set_yscale("log")
    ax.set_xlabel(r"Arrival rate $\lambda$ (customers per hour)")
    ax.set_ylabel(r"$W_q$ in minutes  (log scale)")
    ax.set_title("Expected waiting time vs arrival rate")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 180)

    ax = axes[1]
    ax.plot(lam2, Pw2, color=_COLOR_THEORY, label="M/M/2 (two cashiers)")
    ax.plot(lam3, Pw3, color=_COLOR_ACCENT, label="M/M/3 (three cashiers)")
    ax.axhline(0.05, color=_COLOR_GREY, linestyle=":",
               label="5% service-level red line")
    ax.set_xlabel(r"Arrival rate $\lambda$ (customers per hour)")
    ax.set_ylabel(r"$P(W_q > 2)$")
    ax.set_title(r"Probability the wait exceeds 2 minutes")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.02)

    fig.suptitle("Staffing comparison: M/M/2 vs M/M/3 (each cashier at rate $\\mu = 1$ per minute)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# Driver

def build_all(results_dir: Path, figures_dir: Path) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "validation_metrics.csv"
    pi_path = results_dir / "validation_pi.csv"
    waits_path = results_dir / "waits_at_rho_0p8.csv"
    mm3_path = results_dir / "mm3_comparison.csv"

    metric_rows = _read_metrics_csv(metrics_path)
    pi_data = _read_pi_csv(pi_path)
    waits = _read_waits(waits_path)
    mm3_data = _read_mm3(mm3_path)

    paths: dict[str, Path] = {}

    p = figures_dir / "fig01_rate_diagram.png"
    fig01_rate_diagram(p); paths["fig01"] = p; print(f"  built {p.name}")

    p = figures_dir / "fig02_warmup_convergence.png"
    fig02_warmup_convergence(p); paths["fig02"] = p; print(f"  built {p.name}")

    p = figures_dir / "fig03_stationary_distribution.png"
    fig03_stationary_distribution(pi_data, p); paths["fig03"] = p; print(f"  built {p.name}")

    p = figures_dir / "fig04_core_metrics_vs_rho.png"
    fig04_core_metrics(metric_rows, p); paths["fig04"] = p; print(f"  built {p.name}")

    p = figures_dir / "fig05_erlang_c_vs_rho.png"
    fig05_erlang_c(metric_rows, p); paths["fig05"] = p; print(f"  built {p.name}")

    p = figures_dir / "fig06_waiting_time_cdf.png"
    fig06_waiting_time_cdf(waits, p); paths["fig06"] = p; print(f"  built {p.name}")

    p = figures_dir / "fig07_stress_near_unity.png"
    fig07_stress_near_unity(p); paths["fig07"] = p; print(f"  built {p.name}")

    p = figures_dir / "fig08_pooling_benefit.png"
    fig08_pooling_comparison(p); paths["fig08"] = p; print(f"  built {p.name}")

    p = figures_dir / "fig09_validation_scorecard.png"
    fig09_validation_scorecard(metric_rows, p); paths["fig09"] = p; print(f"  built {p.name}")

    p = figures_dir / "fig10_checkpoint_heatmap.png"
    fig10_checkpoint_heatmap(p); paths["fig10"] = p; print(f"  built {p.name}")

    p = figures_dir / "fig11_staffing_comparison.png"
    fig11_staffing_comparison(mm3_data, p); paths["fig11"] = p; print(f"  built {p.name}")

    return paths


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build all figures for the M/M/2 study.")
    parser.add_argument("--results-dir", type=Path,
                        default=(here.parent / "results"))
    parser.add_argument("--figures-dir", type=Path,
                        default=(here.parent / "figures"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(f"Reading from {args.results_dir}")
    print(f"Writing figures to {args.figures_dir}")
    build_all(args.results_dir, args.figures_dir)
    print("All figures generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
