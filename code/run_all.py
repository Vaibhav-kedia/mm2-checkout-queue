"""Master runner: tests, validation sweep, figures. End-to-end reproduction in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _here() -> Path:
    return Path(__file__).resolve().parent


def _run(label: str, cmd: list[str], cwd: Path) -> float:
    """Run a subprocess, stream output, return elapsed seconds. Exit on failure."""
    print(f"\n=== {label} ===")
    print(f"$ {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"\n[run_all] step failed: {label}  (exit code {proc.returncode})")
        sys.exit(proc.returncode)
    print(f"[run_all] {label} completed in {elapsed:.1f} s")
    return elapsed


# Step runners

def step_tests(code_dir: Path) -> float:
    return _run("Tests (pytest)", [sys.executable, "-m", "pytest", "-q", "tests/"], code_dir)


def step_validation(code_dir: Path, results_dir: Path) -> float:
    return _run(
        "Validation sweep",
        [sys.executable, "validation.py", "--results-dir", str(results_dir)],
        code_dir,
    )


def step_plots(code_dir: Path, results_dir: Path, figures_dir: Path) -> float:
    return _run(
        "Figures",
        [
            sys.executable, "plots.py",
            "--results-dir", str(results_dir),
            "--figures-dir", str(figures_dir),
        ],
        code_dir,
    )


# Driver

def main() -> int:
    here = _here()
    project_root = here.parent
    parser = argparse.ArgumentParser(description="Run tests, validation, and figures end to end.")
    parser.add_argument("--results-dir", type=Path, default=project_root / "results")
    parser.add_argument("--figures-dir", type=Path, default=project_root / "figures")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the pytest step.")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the validation sweep (use existing CSVs).")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    total = 0.0
    if not args.skip_tests:
        total += step_tests(here)
    if not args.skip_validation:
        total += step_validation(here, args.results_dir.resolve())
    if not args.skip_plots:
        total += step_plots(here, args.results_dir.resolve(), args.figures_dir.resolve())

    print(f"\n[run_all] All steps completed. Total wall time: {total:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
