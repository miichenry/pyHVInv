"""
Inversion quality diagnostics for HV-DC joint inversion.

Generates a suite of diagnostic figures to assess the quality and reliability
of the inversion results. Each figure targets a specific aspect of the solution.

Usage
-----
    from hv_dc_invert.quality_matrix import plot_quality_matrix
    plot_quality_matrix(result, obs, save_prefix="quality")

Five figures are produced:
    1. Model ensemble          — spread of accepted models in depth
    2. Marginal distributions  — histogram of each parameter
    3. Parameter correlations  — trade-offs between parameters
    4. Misfit vs parameters    — which parameters are resolved
    5. Convergence diagnostics — misfit history, acceptance rate
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

from .inversion import InversionResult, ObservedCurves
from .model import model_to_depth_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_params(models: list) -> tuple[np.ndarray, list[str]]:
    """
    Flatten a list of (nlayers, 4) model arrays into a 2-D matrix.

    Returns
    -------
    X : ndarray (n_models, n_params)
    names : list of str
        Parameter labels, e.g. ["h1", "Vp1", "Vs1", "rho1", ...]
    """
    n_layers = models[0].shape[0]
    names = []
    for k in range(n_layers):
        suffix = f"{k+1}" if k < n_layers - 1 else "HS"
        if k < n_layers - 1:
            names.append(f"h{suffix}")
        names += [f"Vp{suffix}", f"Vs{suffix}", f"ρ{suffix}"]

    X = []
    for m in models:
        row = []
        for k in range(n_layers):
            if k < n_layers - 1:
                row.append(m[k, 0])   # thickness (skip halfspace)
            row += [m[k, 1], m[k, 2], m[k, 3]]
        X.append(row)

    return np.array(X), names


def _select_models(result: InversionResult, n_best: int, misfit_threshold: Optional[float]):
    """Return indices, models, and misfits for the selected ensemble."""
    misfits = np.array(result.all_misfits)
    if misfit_threshold is not None:
        idx = np.where(misfits <= misfit_threshold)[0]
    else:
        idx = np.argsort(misfits)[:n_best]
    models = [result.all_models[i] for i in idx]
    misfits_sel = misfits[idx]
    return idx, models, misfits_sel


# ---------------------------------------------------------------------------
# Figure 1 — Model ensemble
# ---------------------------------------------------------------------------

def plot_model_ensemble(
    result: InversionResult,
    n_best: int = 100,
    misfit_threshold: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vs and Vp depth profiles for the ensemble of best models.

    How to read
    -----------
    - Tight grey spread → parameter is well-constrained by the data.
    - Wide spread → large uncertainty; the data cannot resolve that depth range.
    - Coloured lines = the single best-fit model.
    """
    _, models, misfits_sel = _select_models(result, n_best, misfit_threshold)

    best = result.best_model
    total_h = np.sum(best[:-1, 0])
    max_depth = total_h * 1.3

    fig, axes = plt.subplots(1, 2, figsize=(8, 7), sharey=True)
    fig.suptitle(
        "Fig 1 — Model Ensemble\n"
        "Spread = uncertainty. Tight = well-constrained. Wide = poorly resolved.",
        fontsize=10, fontweight="bold"
    )

    for k, m in enumerate(models):
        d, vs, vp = model_to_depth_profile(m, max_depth)
        kw = dict(lw=0.6, alpha=0.35)
        axes[0].plot(vs, d, color="0.65", **kw, label="Ensemble" if k == 0 else None)
        axes[1].plot(vp, d, color="lightsteelblue", **kw, label="Ensemble" if k == 0 else None)

    d, vs, vp = model_to_depth_profile(best, max_depth)
    axes[0].plot(vs, d, "r-", lw=2.5, label="Best Vs")
    axes[1].plot(vp, d, "b--", lw=2.0, label="Best Vp")

    for ax, title in zip(axes, ["Vs (m/s)", "Vp (m/s)"]):
        ax.set_xlabel("Velocity (m/s)")
        ax.set_ylim([max_depth, 0])
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Depth (m)")

    fig.text(
        0.5, 0.01,
        f"Showing {len(models)} best models  |  "
        f"Best misfit = {result.best_misfit:.5f}",
        ha="center", fontsize=8, color="0.4"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Marginal distributions
# ---------------------------------------------------------------------------

def plot_marginal_distributions(
    result: InversionResult,
    n_best: int = 500,
    misfit_threshold: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram of each model parameter across the accepted ensemble.

    How to read
    -----------
    - Sharp, peaked histogram → parameter is well-resolved.
    - Flat or multi-modal histogram → parameter is poorly constrained or
      there are multiple acceptable solutions (non-uniqueness).
    - Red dashed line = best-fit value.
    """
    _, models, _ = _select_models(result, n_best, misfit_threshold)
    X, names = _extract_params(models)
    n_params = X.shape[1]

    ncols = 4
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    fig.suptitle(
        "Fig 2 — Marginal Distributions\n"
        "Peaked = well-resolved. Flat or multi-modal = poorly constrained.",
        fontsize=10, fontweight="bold"
    )
    axes = axes.flatten()

    # Best-fit parameter vector
    X_best, _ = _extract_params([result.best_model])
    best_vals = X_best[0]

    for j, (ax, name) in enumerate(zip(axes, names)):
        ax.hist(X[:, j], bins=30, color="steelblue", alpha=0.7, edgecolor="white", lw=0.3)
        ax.axvline(best_vals[j], color="r", lw=1.5, ls="--", label="Best")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for ax in axes[len(names):]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Parameter correlation matrix
# ---------------------------------------------------------------------------

def plot_parameter_correlations(
    result: InversionResult,
    n_best: int = 500,
    misfit_threshold: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Pearson correlation matrix between all model parameters.

    How to read
    -----------
    - Strong red (≈+1) or blue (≈-1) cell → trade-off between those two
      parameters; the data cannot distinguish them independently.
    - Classic seismic trade-off: layer thickness vs velocity (h ↔ Vs).
    - Near-zero (white) → parameters are independent.
    """
    _, models, _ = _select_models(result, n_best, misfit_threshold)
    X, names = _extract_params(models)

    # Remove constant columns (uninformative)
    std = X.std(axis=0)
    keep = std > 0
    X = X[:, keep]
    names = [n for n, k in zip(names, keep) if k]

    corr = np.corrcoef(X.T)

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), max(5, len(names) * 0.6)))
    fig.suptitle(
        "Fig 3 — Parameter Correlation Matrix\n"
        "Red = positive trade-off. Blue = negative trade-off. White = independent.",
        fontsize=10, fontweight="bold"
    )

    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(names)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Misfit vs parameters
# ---------------------------------------------------------------------------

def plot_misfit_vs_parameters(
    result: InversionResult,
    n_best: int = 500,
    misfit_threshold: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of misfit vs each model parameter.

    How to read
    -----------
    - Clear V-shape (minimum visible) → parameter is resolved; the inversion
      found a meaningful optimum.
    - Flat cloud (no minimum) → parameter is unconstrained; the data has no
      sensitivity to it.
    - Red dashed line = best-fit value.
    """
    _, models, misfits = _select_models(result, n_best, misfit_threshold)
    X, names = _extract_params(models)
    n_params = X.shape[1]

    X_best, _ = _extract_params([result.best_model])
    best_vals = X_best[0]

    ncols = 4
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    fig.suptitle(
        "Fig 4 — Misfit vs Parameters\n"
        "V-shape = parameter resolved. Flat cloud = parameter unconstrained.",
        fontsize=10, fontweight="bold"
    )
    axes = axes.flatten()

    for j, (ax, name) in enumerate(zip(axes, names)):
        sc = ax.scatter(X[:, j], misfits, c=misfits, cmap="viridis_r",
                        s=4, alpha=0.5, linewidths=0)
        ax.axvline(best_vals[j], color="r", lw=1.5, ls="--", label="Best")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Misfit", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    for ax in axes[len(names):]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Convergence diagnostics
# ---------------------------------------------------------------------------

def plot_convergence(
    result: InversionResult,
    window: int = 100,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three convergence panels: misfit history, misfit distribution, acceptance rate.

    How to read
    -----------
    Misfit history:
        Should flatten out (plateau) well before the end of the run.
        Still falling at the end → run more iterations.

    Misfit histogram:
        Sharp drop-off toward low misfits = well-defined solution.
        Broad distribution = solution is non-unique or poorly constrained.

    Acceptance rate:
        For Metropolis MC: should be 20–50% on average. Very low (<5%) means
        the perturbation step is too large; very high (>80%) means it's too small.
        For SA: rate naturally decreases as temperature drops — that is expected.
    """
    history = np.array(result.misfit_history)
    all_misfits = np.array(result.all_misfits)
    finite_misfits = all_misfits[np.isfinite(all_misfits)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        "Fig 5 — Convergence Diagnostics\n"
        "History: plateau = converged. Histogram: sharp drop = well-defined. "
        "Acceptance: 20–50% ideal for MC.",
        fontsize=10, fontweight="bold"
    )

    # --- Panel A: misfit history ---
    ax = axes[0]
    finite_hist = history[np.isfinite(history)]
    ax.semilogy(finite_hist, color="steelblue", lw=1.0, alpha=0.8, label="Best misfit")
    ax.set_xlabel("Model evaluations")
    ax.set_ylabel("Best misfit (log scale)")
    ax.set_title("A — Misfit History")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Running minimum annotation
    if len(finite_hist) > 10:
        mid = len(finite_hist) // 2
        pct_drop = 100 * (finite_hist[mid] - finite_hist[-1]) / (finite_hist[mid] + 1e-12)
        ax.text(0.97, 0.97,
                f"Drop (2nd half): {pct_drop:.1f}%\n"
                f"{'Converged' if pct_drop < 5 else 'May need more iterations'}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="darkred" if pct_drop >= 5 else "darkgreen",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # --- Panel B: misfit histogram ---
    ax = axes[1]
    ax.hist(finite_misfits, bins=50, color="steelblue", alpha=0.7,
            edgecolor="white", lw=0.3)
    ax.axvline(result.best_misfit, color="r", lw=2, ls="--", label=f"Best = {result.best_misfit:.4f}")
    ax.set_xlabel("Misfit")
    ax.set_ylabel("Count")
    ax.set_title("B — Misfit Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel C: rolling acceptance rate ---
    ax = axes[2]
    # Acceptance = misfit improved or model was accepted (proxied by misfit change)
    # We approximate acceptance from the misfit history: accepted if misfit <= previous best
    if len(history) > window:
        accepted = np.zeros(len(history), dtype=float)
        for i in range(1, len(history)):
            accepted[i] = 1.0 if history[i] <= history[i - 1] else 0.0
        roll = np.convolve(accepted, np.ones(window) / window, mode="valid")
        x = np.arange(window - 1, len(history))
        ax.plot(x, roll * 100, color="steelblue", lw=1.2)
        ax.axhspan(20, 50, color="green", alpha=0.12, label="Ideal range (20–50%)")
        ax.axhline(20, color="green", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(50, color="green", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("Model evaluations")
        ax.set_ylabel("Acceptance rate (%)")
        ax.set_ylim([0, 100])
        ax.set_title(f"C — Rolling Acceptance Rate\n(window = {window})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Not enough iterations\nto compute rolling rate",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("C — Rolling Acceptance Rate")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_quality_matrix(
    result: InversionResult,
    obs: ObservedCurves,
    n_best: int = 100,
    n_best_stats: int = 500,
    misfit_threshold: Optional[float] = None,
    save_prefix: Optional[str] = None,
    show: bool = True,
) -> list[plt.Figure]:
    """
    Generate all five quality diagnostic figures.

    Parameters
    ----------
    result : InversionResult
    obs : ObservedCurves
    n_best : int
        Number of best models for the ensemble profile plot (Fig 1).
    n_best_stats : int
        Number of best models for statistical plots (Figs 2–4).
        Use a larger number for better statistics.
    misfit_threshold : float, optional
        If given, use all models below this misfit threshold instead of n_best.
    save_prefix : str, optional
        If given, save figures as "<prefix>_fig1.png", ..., "<prefix>_fig5.png".
    show : bool
        Whether to call plt.show() at the end.

    Returns
    -------
    figs : list of 5 matplotlib Figures
    """
    def _path(n):
        return f"{save_prefix}_fig{n}.png" if save_prefix else None

    figs = [
        plot_model_ensemble(result, n_best, misfit_threshold, _path(1)),
        plot_marginal_distributions(result, n_best_stats, misfit_threshold, _path(2)),
        plot_parameter_correlations(result, n_best_stats, misfit_threshold, _path(3)),
        plot_misfit_vs_parameters(result, n_best_stats, misfit_threshold, _path(4)),
        plot_convergence(result, save_path=_path(5)),
    ]

    if show:
        plt.show()

    return figs
