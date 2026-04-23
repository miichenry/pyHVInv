#!/usr/bin/env python3
"""
HV-DC Joint Inversion Workflow
================================
Python port of HV-INV (Garcia-Jerez et al. 2016; Piña-Flores et al. 2017).

Loads observed H/V and/or dispersion curve data, then inverts for a 1D
layered earth model using the Diffuse Field Assumption (DFA) forward model.

Forward calculations use the HVf compiled executable (FORTRAN).

Usage
-----
Edit the CONFIGURATION section below, then run:

    python run_inversion.py

Input file formats
------------------
H/V file  (2 or 3 columns, space/tab separated):
    frequency(Hz)   HV_amplitude   [HV_std_optional]

Dispersion curve file (2 or 3 columns):
    frequency(Hz)   velocity(m/s)   [velocity_std_optional]

References
----------
Garcia-Jerez A., Piña-Flores J., Sánchez-Sesma F.J., Luzón F., Perton M.
(2016) A computer code for forward calculation and inversion of the H/V
spectral ratio under the diffuse field assumption, Computers & Geosciences
97, 67–78.

Piña-Flores J., Perton M., Garcia-Jerez A., Carmona E., Luzón F.,
Molina-Villegas J.C., Sánchez-Sesma F.J. (2017). The inversion of spectral
ratio H/V in a layered system using the Diffuse Field Assumption (DFA),
Geophysical Journal International 208, 577–588.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add package to path if running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from hv_dc_invert.forward import compute_hv, compute_dc
from hv_dc_invert.inversion import (
    ObservedCurves, run_random_search, run_monte_carlo, run_simulated_annealing
)
from hv_dc_invert.model import make_bounds, validate_and_fix_bounds
from hv_dc_invert.plot import plot_results, print_summary
from hv_dc_invert.quality_matrix import plot_quality_matrix

# =============================================================================
#  CONFIGURATION — Edit this section for your dataset
# =============================================================================

# ---- Data files ----
HV_FILE = "apollo/hvsr_38.3797_14.97612.dat"           # set to None to skip
DC_FILE = "apollo/disp_38.3797_14.97612_td.dat"  # set to None to skip
# NOTE: DC file must be in frequency (Hz) and velocity (m/s); convert if your
#       source data uses period (s) and/or km/s before using. I added a quick conversion in the code for 
# the provided example files, which are in period and km/s. 
# If your files are already in the correct units, you can remove 
# that conversion step in the code (search for "dc_freq, dc_vel, dc_std = 1.0 / dc_freq, dc_vel * 1000.0, dc_std * 1000.0" and comment it out).

# ---- HVf executable path ----
# Auto-detect OS; override here if needed:
import platform
_OS = platform.system()
if _OS == "Darwin":
    HVF_EXE = "SIMPAC-2024-195/exe/exe_Mac/HVf"
elif _OS == "Linux":
    HVF_EXE = "SIMPAC-2024-195/exe/exe_Linux/HVf"
else:  # Windows
    HVF_EXE = "SIMPAC-2024-195/exe/exe_Win/HVf.exe"

# ---- Dispersion curve type ----
DC_POLARIZATION = "rayleigh"   # 'rayleigh' or 'love'
DC_VELOCITY_TYPE = "group"     # 'phase' or 'group'
DC_MODE = 0                    # 0 = fundamental mode

# ---- Model parameterization ----
# Define bounds for each layer (including halfspace).
# Last layer = halfspace: set h_min=h_max=0.
#
# 4-layer example (3 layers + halfspace):
N_LAYERS = 4

#          Layer 1   Layer 2   Layer 3   Halfspace                                                       
H_MIN  = [  1,        50,       400,      0]
H_MAX  = [100,       450,       650,      0]                                                             
                                                                                                        
VP_MIN = [ 200,      500,      1500,   1500]                                                             
VP_MAX = [1500,     3000,      4000,   5000]                                                             
                                                                                                        
VS_MIN = [ 100,      300,       800,    800]                                                             
VS_MAX = [1000,     1500,      2500,   3000]
                                                                                                        
RHO_MIN = [1500,    1800,      1800,   1800]                                                            
RHO_MAX = [2500,    2800,      2800,   2800]                                                             
                                                                                                        
NU_MIN  = [0.25,    0.25,      0.25,   0.25]
NU_MAX  = [0.45,    0.45,      0.45,   0.45] 

ALLOW_LVZ_VS = True    # allow low-velocity zones in Vs
ALLOW_LVZ_VP = True    # allow low-velocity zones in Vp22

# ---- Forward model settings ----
N_RAYLEIGH_MODES = 5    # Rayleigh modes for HV calculation
N_LOVE_MODES = 5        # Love modes for HV calculation
N_WAVENUMBERS_MIN = 50   # initial integration points for body waves (0 = skip body waves)
N_WAVENUMBERS_MAX = 100  # maximum integration points for body waves (incremented each forward call)
APSV = 1e-4             # imaginary part of freq. to stabilize PSV (body waves)

# ---- Inversion settings ----
RANDOM_SEED = 42        # set to None for non-reproducible runs

# Stage 1: initial random search
N_RANDOM = 500          # number of random models in initial search

# Stage 2: choose method — 'montecarlo' or 'sa' (simulated annealing)
INVERSION_METHOD = "montecarlo"

# Monte Carlo / SA shared
N_ITERATIONS = 5000     # iterations per stage
PERTURBATION_PCT = 5.0  # perturbation range (% of each parameter's range)

# SA-specific
SA_N_TEMPERATURES = 10  # number of temperature reduction steps
SA_N_FINAL = 500        # iterations for final Markov chain
SA_COOLING_RATE = 0.90  # multiplicative cooling factor

# Joint inversion weight
EQUALIZE_WEIGHTS = True  # auto-equalize HV/DC by number of samples
DC_WEIGHT = 0.5          # manual weight for DC (used if EQUALIZE_WEIGHTS=False)

# ---- Standard deviation (when not in data file) ----
# Used to normalize the misfit; matched to HV-INV's default (10% of amplitude).
DEFAULT_STD_PCT = 10.0   # % of measurement amplitude
MAX_CURVE_SAMPLES = 100  # max samples for HV and DC curves (None to keep original)

# ---- Output ----
OUTPUT_DIR = "results"                  # directory for all output files (created if needed)
SAVE_FIGURE = "inversion_results.png"   # set to None to skip saving
SHOW_PLOTS  = False                      # set to False to skip plt.show() (batch/headless)
N_BEST_PLOT = 100                       # ensemble of N best models to show

# =============================================================================
#  END OF CONFIGURATION
# =============================================================================


def load_data(filepath: str, default_std_pct: float = 10.0):
    """
    Load 2- or 3-column whitespace-separated data file.

    If no std column is present, std defaults to `default_std_pct`% of the
    measured amplitude (matching HV-INV Zombie-mode behaviour).

    Parameters
    ----------
    filepath : str
    default_std_pct : float
        Percent401e of measured value used as std when the file has no std column.

    Returns
    -------
    freq, values, std : ndarray
    """
    data = np.loadtxt(filepath, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    freq = data[:, 0]
    values = data[:, 1]
    if data.shape[1] >= 3:
        std = data[:, 2]
    else:
        std = values * (default_std_pct / 100.0)
    return freq, values, std


def save_best_model(result, output_dir: Path) -> Path:
    """
    Save the best-fit model to a .txt file in output_dir.

    Format matches HV-INV output:
        n_layers
        thickness(m)  Vp(m/s)  Vs(m/s)  density(kg/m3)  Poisson
    Last row is the halfspace (thickness = 0).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "best_model.txt"

    model = result.best_model
    n_layers = model.shape[0]

    with open(out_path, "w") as f:
        f.write(f"# Best-fit model  |  misfit = {result.best_misfit:.6f}\n")
        f.write(f"# {'Layer':<8} {'Thickness(m)':<15} {'Vp(m/s)':<12} "
                f"{'Vs(m/s)':<12} {'Density(kg/m3)':<16} {'Poisson'}\n")
        f.write(f"{n_layers}\n")
        for k in range(n_layers):
            h, vp, vs, rho = model[k]
            nu = 0.5 * (vp**2 - 2*vs**2) / (vp**2 - vs**2) if vp > vs else float("nan")
            label = "halfspace" if k == n_layers - 1 else str(k + 1)
            f.write(f"# {label:<8} ")
            f.write(f"{h:<15.4f} {vp:<12.4f} {vs:<12.4f} {rho:<16.4f} {nu:.6f}\n")
            f.write(f"{h:.4f}\t{vp:.4f}\t{vs:.4f}\t{rho:.4f}\n")

    print(f"Best model saved to: {out_path}")

    summary = {
        "misfit": float(result.best_misfit),
        "n_layers": int(n_layers),
        "layers": [
            {
                "h":   float(model[k, 0]),
                "vp":  float(model[k, 1]),
                "vs":  float(model[k, 2]),
                "rho": float(model[k, 3]),
                "nu":  float(0.5 * (model[k,1]**2 - 2*model[k,2]**2)
                             / (model[k,1]**2 - model[k,2]**2))
                       if model[k, 1] > model[k, 2] else None,
            }
            for k in range(n_layers)
        ],
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return out_path


def make_forward_function(hvf_exe, fwd_cfg):
    """
    Return a forward function: f(model, obs) -> (hv_syn, dc_syn).
    hv_syn or dc_syn is None if that curve type is not requested.

    n_wavenumbers is incremented by 1 each call (matching HV-INV behaviour),
    capped at n_wavenumbers_max.
    """
    # Use a list so the closure can mutate it
    nks = [fwd_cfg["n_wavenumbers"]]
    nks_max = fwd_cfg["n_wavenumbers_max"]

    def forward(model, obs):
        hv_syn = None
        dc_syn = None

        if obs.has_hv:
            hv_syn = compute_hv(
                model, obs.hv_freq, hvf_exe,
                n_rayleigh=fwd_cfg["n_rayleigh"],
                n_love=fwd_cfg["n_love"],
                n_wavenumbers=nks[0],
                apsv=fwd_cfg["apsv"],
            )
            nks[0] = min(nks[0] + 1, nks_max)

        if obs.has_dc:
            dc_syn = compute_dc(
                model, obs.dc_freq, hvf_exe,
                polarization=obs.dc_polarization,
                velocity_type=obs.dc_velocity_type,
                mode=obs.dc_mode,
            )

        return hv_syn, dc_syn

    return forward


def main():
    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)

    # Resolve paths relative to this script's location
    base = Path(__file__).parent

    # Make HVf executable
    hvf_path = str(base / HVF_EXE)
    if not os.path.isfile(hvf_path):
        raise FileNotFoundError(f"HVf executable not found: {hvf_path}")
    os.chmod(hvf_path, 0o755)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    obs = ObservedCurves(
        dc_polarization=DC_POLARIZATION,
        dc_velocity_type=DC_VELOCITY_TYPE,
        dc_mode=DC_MODE,
    )

    if HV_FILE:
        hv_path = str(base / HV_FILE)
        hv_freq, hv_amp, hv_std = load_data(hv_path, DEFAULT_STD_PCT)
        if MAX_CURVE_SAMPLES is not None and len(hv_freq) > MAX_CURVE_SAMPLES:                  
            freq_new = np.logspace(np.log10(hv_freq.min()), np.log10(hv_freq.max()), MAX_CURVE_SAMPLES)                                                                                 
            hv_amp = np.interp(freq_new, hv_freq, hv_amp)                                
            hv_std = np.interp(freq_new, hv_freq, hv_std)                                
            hv_freq = freq_new                                                           
            print(f"HV resampled to {MAX_CURVE_SAMPLES} points (log-spaced)") 
        obs.hv_freq = hv_freq
        obs.hv_amp = hv_amp
        obs.hv_std = hv_std
        print(f"Loaded HV data: {len(hv_freq)} points  "
              f"[{hv_freq[0]:.3f} – {hv_freq[-1]:.3f} Hz]")

    if DC_FILE:
        dc_path = str(base / DC_FILE)
        dc_freq, dc_vel, dc_std = load_data(dc_path, DEFAULT_STD_PCT)
        dc_freq, dc_vel, dc_std = 1.0 / dc_freq, dc_vel * 1000.0, dc_std * 1000.0 # Convert from period (s) to frequency (Hz) and from km/s to m/s
        if MAX_CURVE_SAMPLES is not None and len(dc_freq) > MAX_CURVE_SAMPLES:           
            freq_new = np.logspace(np.log10(dc_freq.min()), np.log10(dc_freq.max()), MAX_CURVE_SAMPLES)
            dc_vel = np.interp(freq_new, dc_freq, dc_vel)
            dc_std = np.interp(freq_new, dc_freq, dc_std)
            dc_freq = freq_new
            print(f"DC resampled to {MAX_CURVE_SAMPLES} points (log-spaced)")
        obs.dc_freq = dc_freq
        obs.dc_vel = dc_vel
        obs.dc_std = dc_std
        print(f"Loaded DC data: {len(dc_freq)} points  "
              f"[{dc_freq[0]:.3f} – {dc_freq[-1]:.3f} Hz]")

    if not obs.has_hv and not obs.has_dc:
        raise ValueError("No data loaded. Set HV_FILE and/or DC_FILE.")

    # ------------------------------------------------------------------
    # 2. Build parameter bounds
    # ------------------------------------------------------------------
    var = make_bounds(
        N_LAYERS,
        H_MIN, H_MAX,
        VP_MIN, VP_MAX,
        VS_MIN, VS_MAX,
        RHO_MIN, RHO_MAX,
        NU_MIN, NU_MAX,
    )
    print("\nChecking parameter bounds compatibility...")                                       
    var = validate_and_fix_bounds(var, allow_lvz_vs=ALLOW_LVZ_VS, allow_lvz_vp=ALLOW_LVZ_VP)

    # ------------------------------------------------------------------
    # 3. Forward function
    # ------------------------------------------------------------------
    fwd_cfg = dict(
        n_rayleigh=N_RAYLEIGH_MODES,
        n_love=N_LOVE_MODES,
        n_wavenumbers=N_WAVENUMBERS_MIN,
        n_wavenumbers_max=N_WAVENUMBERS_MAX,
        apsv=APSV,
    )
    forward_fn = make_forward_function(hvf_path, fwd_cfg)

    # Quick sanity check with mid-range model
    print("\nRunning forward model sanity check...")
    test_model = np.column_stack([
        np.array(H_MIN) + (np.array(H_MAX) - np.array(H_MIN)) / 2,
        np.array(VP_MIN) + (np.array(VP_MAX) - np.array(VP_MIN)) / 2,
        np.array(VS_MIN) + (np.array(VS_MAX) - np.array(VS_MIN)) / 2,
        np.array(RHO_MIN) + (np.array(RHO_MAX) - np.array(RHO_MIN)) / 2,
    ])
    test_model[-1, 0] = 0.0  # halfspace
    hv_test, dc_test = forward_fn(test_model, obs)
    if obs.has_hv:
        print(f"  HV forward test: {'OK' if hv_test is not None else 'FAILED'}")
    if obs.has_dc:
        print(f"  DC forward test: {'OK' if dc_test is not None else 'FAILED'}")

    # ------------------------------------------------------------------
    # 4. Progress callback
    # ------------------------------------------------------------------
    def progress(n_eval, best_misfit, *args):
        if n_eval % 100 == 0:
            print(f" [{n_eval:5d} models]  Best misfit = {best_misfit:.5f}")

    # ------------------------------------------------------------------
    # 5. Stage 1 – Initial random search
    # ------------------------------------------------------------------
    print(f"\n--- Stage 1: Random Search ({N_RANDOM} models) ---")
    random_result = run_random_search(
        var=var,
        obs=obs,
        forward_fn=forward_fn,
        n_models=N_RANDOM,
        allow_lvz_vs=ALLOW_LVZ_VS,
        allow_lvz_vp=ALLOW_LVZ_VP,
        equalize_weights=EQUALIZE_WEIGHTS,
        dc_weight=DC_WEIGHT,
        rng=rng,
        callback=progress,
    )
    print(f"  Best misfit after random search: {random_result.best_misfit:.5f}")

    # ------------------------------------------------------------------
    # 6. Stage 2 – Iterative inversion
    # ------------------------------------------------------------------
    if INVERSION_METHOD == "montecarlo":
        print(f"\n--- Stage 2: Monte Carlo ({N_ITERATIONS} iterations) ---")
        final_result = run_monte_carlo(
            initial_result=random_result,
            var=var,
            obs=obs,
            forward_fn=forward_fn,
            n_iterations=N_ITERATIONS,
            perturbation_pct=PERTURBATION_PCT,
            allow_lvz_vs=ALLOW_LVZ_VS,
            allow_lvz_vp=ALLOW_LVZ_VP,
            equalize_weights=EQUALIZE_WEIGHTS,
            dc_weight=DC_WEIGHT,
            rng=rng,
            callback=progress,
        )

    elif INVERSION_METHOD == "sa":
        print(f"\n--- Stage 2: Simulated Annealing "
              f"({SA_N_TEMPERATURES} temp steps × {N_ITERATIONS} iters) ---")
        final_result = run_simulated_annealing(
            initial_result=random_result,
            var=var,
            obs=obs,
            forward_fn=forward_fn,
            n_iterations=N_ITERATIONS,
            n_temperatures=SA_N_TEMPERATURES,
            n_final_iter=SA_N_FINAL,
            perturbation_pct=PERTURBATION_PCT,
            cooling_rate=SA_COOLING_RATE,
            allow_lvz_vs=ALLOW_LVZ_VS,
            allow_lvz_vp=ALLOW_LVZ_VP,
            equalize_weights=EQUALIZE_WEIGHTS,
            dc_weight=DC_WEIGHT,
            rng=rng,
            callback=progress,
        )
    else:
        raise ValueError(f"Unknown method: {INVERSION_METHOD!r}. Use 'montecarlo' or 'sa'.")

    print(f"  Final best misfit: {final_result.best_misfit:.5f}")

    # ------------------------------------------------------------------
    # 7. Results
    # ------------------------------------------------------------------
    output_dir = base / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(final_result, obs)
    save_best_model(final_result, output_dir)

    mode_str = "" if DC_MODE == 0 else f" mode-{DC_MODE}"
    fig_title = (
        f"HV-DC Joint Inversion  —  {INVERSION_METHOD.upper()}"
        f"\nBest misfit = {final_result.best_misfit:.5f}  |  "
        f"{final_result.n_evaluated} models evaluated"
    )
    fig = plot_results(
        final_result,
        obs,
        n_best=N_BEST_PLOT,
        title=fig_title,
        save_path=str(output_dir / SAVE_FIGURE) if SAVE_FIGURE else None,
    )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")

    plot_quality_matrix(
        final_result, obs,
        save_prefix=str(output_dir / "quality") if SAVE_FIGURE else None,
        show=SHOW_PLOTS,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HV-DC Joint Inversion")
    parser.add_argument("--hv", default=None, help="Override HV_FILE")
    parser.add_argument("--dc", default=None, help="Override DC_FILE")
    parser.add_argument("--output-dir", default=None, help="Override OUTPUT_DIR")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() (batch/headless)")
    args = parser.parse_args()
    if args.hv is not None:
        HV_FILE = args.hv
    if args.dc is not None:
        DC_FILE = args.dc
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    if args.no_show:
        SHOW_PLOTS = False
    main()
