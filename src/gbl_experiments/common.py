"""Shared helpers: paths, plotting style, integration of the optimizer ODE, results output."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

RTOL, ATOL = 1e-11, 1e-13

# Set by seeds.py: WRITE = False skips figures and result files; every failed check is appended to FAILURES.
WRITE = True
FAILURES: list[str] = []


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def save_figure(fig, name: str) -> Path | None:
    if not WRITE:
        plt.close(fig)
        return None
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def save_results(name: str, results: dict) -> None:
    if not WRITE:
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(f"  wrote {path.relative_to(ROOT)}")


def report(name: str, value: float, tol: float | None = None) -> float:
    flag = "" if tol is None else ("  [OK]" if value < tol else "  [FAIL]")
    if flag == "  [FAIL]":
        FAILURES.append(f"{name}: {value:.3e} (tolerance {tol:.0e})")
    print(f"  {name:<70s} {value:.3e}{flag}")
    return float(value)


def integrate(rhs, y0, t0: float, t1: float, n_eval: int = 400, log_times: bool = False):
    """Integrate dy/dt = rhs(t, y) with a high-order adaptive method and tight tolerances."""
    if log_times:
        t_eval = np.geomspace(t0, t1, n_eval)
    else:
        t_eval = np.linspace(t0, t1, n_eval)
    t_eval[0], t_eval[-1] = t0, t1
    sol = solve_ivp(rhs, (t0, t1), np.asarray(y0, dtype=float), method="DOP853",
                    t_eval=t_eval, rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


def damped_rhs(grad, mu, gamma):
    """First-order form of the optimizer ODE mu(t) x'' + gamma(t) x' + grad L(x) = 0, state (x, v).

    mu, gamma are callables of t; mu = None means gradient flow, gamma(t) x' = -grad L(x)."""
    def rhs(t, y):
        if mu is None:
            return -np.asarray(grad(y)) / gamma(t)
        d = y.size // 2
        x, v = y[:d], y[d:]
        return np.concatenate([v, -(gamma(t) * v + np.asarray(grad(x))) / mu(t)])
    return rhs


def rk4(f, y0, t0: float, t1: float, n_steps: int, n_save: int = 400, log_times: bool = False):
    """Fixed-step classical Runge-Kutta for dy/dt = f(t, y), with f a JAX function; jitted via lax.scan.

    With log_times, the step is uniform in u = log t (dy/du = t f(t, y)), which suits gamma = 3/t."""
    import jax.numpy as jnp

    if log_times:
        u0, u1 = np.log(t0), np.log(t1)
        g = lambda u, y: jnp.exp(u) * f(jnp.exp(u), y)  # noqa: E731
    else:
        u0, u1, g = t0, t1, f
    du = (u1 - u0) / n_steps
    every = max(n_steps // n_save, 1)

    def step(carry, _):
        u, y = carry
        k1 = g(u, y)
        k2 = g(u + du / 2, y + du / 2 * k1)
        k3 = g(u + du / 2, y + du / 2 * k2)
        k4 = g(u + du, y + du * k3)
        return (u + du, y + du / 6 * (k1 + 2 * k2 + 2 * k3 + k4)), None

    def block(carry, _):
        carry, _ = jax.lax.scan(step, carry, None, length=every)
        return carry, carry

    @jax.jit
    def go(y0):
        y0 = jnp.asarray(y0, dtype=jnp.float64)
        _, (us, ys) = jax.lax.scan(block, (jnp.float64(u0), y0), None, length=n_steps // every)
        return jnp.concatenate([jnp.array([u0]), us]), jnp.concatenate([y0[None], ys])

    us, ys = go(y0)
    us, ys = np.asarray(us), np.asarray(ys)
    return (np.exp(us) if log_times else us), ys
