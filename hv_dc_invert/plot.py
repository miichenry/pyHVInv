"""
Visualization for HV-DC inversion results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional
from .inversion import InversionResult, ObservedCurves
from .model import model_to_depth_profile


def plot_results(
    result: InversionResult,
    obs: ObservedCurves,
    n_best: int = 50,
    misfit_threshold: Optional[float] = None,
    figsize: tuple = (14, 8),
    title: str = "HV-DC Joint Inversion Results",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot inversion results: velocity profile, HV fit, DC fit, misfit history.

    Parameters
    ----------
    result : InversionResult
    obs : ObservedCurves
    n_best : int
        Number of best models to show as ensemble (gray).
    misfit_threshold : float, optional
        If given, show all models with misfit < threshold instead of n_best.
    figsize : tuple
    title : str
    save_path : str, optional
        If given, save figure to this path.

    Returns
    -------
    fig : matplotlib Figure
    """
    n_panels = 1  # velocity profile always shown
    if obs.has_hv:
        n_panels += 1
    if obs.has_dc:
        n_panels += 1
    n_panels += 1  # misfit history

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(1, n_panels, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]

    # --- Select ensemble of models to show ---
    misfits = np.array(result.all_misfits)
    if len(misfits) == 0:
        return fig

    if misfit_threshold is not None:
        idx = np.where(misfits <= misfit_threshold)[0]
    else:
        idx = np.argsort(misfits)[:n_best]

    # Max depth for profile plot
    best = result.best_model
    total_thickness = np.sum(best[:-1, 0])
    max_depth = total_thickness * 1.3

    ax_idx = 0

    # ---- Panel 1: Velocity profile ----
    ax = axes[ax_idx]
    ax_idx += 1

    # Plot ensemble (Vs and Vp in grey)
    for k, i in enumerate(idx):
        m = result.all_models[i]
        d, vs, vp = model_to_depth_profile(m, max_depth)
        label_vs = "Ensemble Vs" if k == 0 else None
        label_vp = "Ensemble Vp" if k == 0 else None
        ax.plot(vs, d, color="0.75", lw=0.5, alpha=0.4, label=label_vs)
        ax.plot(vp, d, color="lightblue", lw=0.5, alpha=0.4, label=label_vp)

    # Plot best model Vs and Vp
    d, vs, vp = model_to_depth_profile(best, max_depth)
    ax.plot(vs, d, "r-", lw=2.5, label="Best Vs")
    ax.plot(vp, d, "b--", lw=1.5, label="Best Vp", alpha=0.7)

    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_ylim([max_depth, 0])
    ax.legend(fontsize=8)
    ax.set_title("Velocity Profile")
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: H/V fit (if applicable) ----
    if obs.has_hv:
        ax = axes[ax_idx]
        ax_idx += 1

        # Ensemble
        for i in idx:
            if i < len(result.all_hv) and result.all_hv[i] is not None:
                ax.plot(obs.hv_freq, result.all_hv[i], color="0.75", lw=0.5, alpha=0.4)

        # Observed
        if obs.hv_std is not None and not np.all(obs.hv_std == 1):
            ax.fill_between(
                obs.hv_freq,
                obs.hv_amp - obs.hv_std,
                obs.hv_amp + obs.hv_std,
                color="steelblue", alpha=0.3, label="±1σ"
            )
        ax.plot(obs.hv_freq, obs.hv_amp, "b-", lw=2, label="Observed HV")

        # Best fit
        if result.best_hv is not None:
            ax.plot(obs.hv_freq, result.best_hv, "r-", lw=2.5, label="Best fit")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("H/V Amplitude")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.set_title("H/V Spectral Ratio")
        ax.grid(True, which="both", alpha=0.3)

    # ---- Panel 3: Dispersion curve fit (if applicable) ----
    if obs.has_dc:
        ax = axes[ax_idx]
        ax_idx += 1

        # Ensemble
        for i in idx:
            if i < len(result.all_dc) and result.all_dc[i] is not None:
                ax.plot(obs.dc_freq, result.all_dc[i], color="0.75", lw=0.5, alpha=0.4)

        # Observed
        if obs.dc_std is not None and not np.all(obs.dc_std == 1):
            ax.fill_between(
                obs.dc_freq,
                obs.dc_vel - obs.dc_std,
                obs.dc_vel + obs.dc_std,
                color="darkorange", alpha=0.3, label="±1σ"
            )
        ax.plot(obs.dc_freq, obs.dc_vel, color="darkorange", lw=2, label="Observed DC")

        # Best fit
        if result.best_dc is not None:
            ax.plot(obs.dc_freq, result.best_dc, "r-", lw=2.5, label="Best fit")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase Velocity (m/s)")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        pol_label = obs.dc_polarization.capitalize()
        vel_label = obs.dc_velocity_type.capitalize()
        ax.set_title(f"Dispersion Curve\n({pol_label} {vel_label}, mode {obs.dc_mode})")
        ax.grid(True, which="both", alpha=0.3)

    # ---- Panel 4: Misfit history ----
    ax = axes[ax_idx]
    history = np.array(result.misfit_history)
    history = history[np.isfinite(history)]
    ax.semilogy(history, "r-", lw=1.5)
    ax.set_xlabel("Model evaluations")
    ax.set_ylabel("Best misfit (log scale)")
    ax.set_title("Misfit History")
    ax.grid(True, which="both", alpha=0.3)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception:
            plt.subplots_adjust(wspace=0.35, top=0.90)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def print_summary(result: InversionResult, obs: ObservedCurves) -> None:
    """Print a text summary of the inversion result."""
    model = result.best_model
    n_layers = model.shape[0]

    print("\n" + "=" * 60)
    print("  INVERSION SUMMARY")
    print("=" * 60)
    print(f"  Models evaluated : {result.n_evaluated}")
    print(f"  Valid models     : {result.n_valid}")
    print(f"  Minimum misfit   : {result.best_misfit:.6f}")
    print(f"  Elapsed time     : {result.elapsed_time:.1f} s")
    print()
    print("  BEST-FIT MODEL")
    print(f"  {'Layer':<6} {'Thickness(m)':<14} {'Vp(m/s)':<12} {'Vs(m/s)':<12} {'Density(kg/m3)':<15} {'Poisson':<8}")
    print("  " + "-" * 65)
    for k in range(n_layers):
        h = model[k, 0]
        vp = model[k, 1]
        vs = model[k, 2]
        rho = model[k, 3]
        nu = 0.5 * (vp**2 - 2*vs**2) / (vp**2 - vs**2) if vp > vs else float('nan')
        label = "halfspace" if k == n_layers - 1 else str(k + 1)
        h_str = "0 (HS)" if k == n_layers - 1 else f"{h:.2f}"
        print(f"  {label:<6} {h_str:<14} {vp:<12.1f} {vs:<12.1f} {rho:<15.1f} {nu:<8.3f}")
    print("=" * 60 + "\n")
