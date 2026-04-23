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
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add package to path if running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from hv_dc_invert.forward import compute_hv, compute_dc
from hv_dc_invert.inversion import (
    ObservedCurves, run_random_search, run_monte_carlo, run_simulated_annealing
)
from hv_dc_invert.model import make_bounds
from hv_dc_invert.plot import plot_results, print_summary

# =============================================================================
#  CONFIGURATION — Edit this section for your dataset
# =============================================================================

# ---- Data files ----
HV_FILE = "apollo/hvsr_38.3797_14.97612.dat"           # set to None to skip
DC_FILE = "apollo/disp_38.3797_14.97612_td.dat"  # set to None to skip
# NOTE: DC file must be in frequency (Hz) and velocity (m/s); convert if your
#       source data uses period (s) and/or km/s before using.

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
# 3-layer example (2 layers + halfspace):
N_LAYERS = 3

#         Layer 1  Layer 2  Halfspace
H_MIN  = [  1,      10,      0  ]   # min thickness (m)
H_MAX  = [ 30,      80,      0  ]   # max thickness (m)

VP_MIN = [ 200,    500,    1500 ]   # min Vp (m/s)
VP_MAX = [1500,   3000,    4000 ]   # max Vp (m/s)

VS_MIN = [ 100,    200,     600 ]   # min Vs (m/s)
VS_MAX = [ 600,   1200,    2500 ]   # max Vs (m/s)

RHO_MIN = [1500,  1800,    2000 ]   # min density (kg/m³)
RHO_MAX = [2000,  2500,    2500 ]   # max density (kg/m³)

NU_MIN  = [0.20,  0.20,    0.20 ]   # min Poisson's ratio
NU_MAX  = [0.45,  0.45,    0.45 ]   # max Poisson's ratio

ALLOW_LVZ_VS = True    # allow low-velocity zones in Vs
ALLOW_LVZ_VP = True    # allow low-velocity zones in Vp

# ---- Forward model settings ----
N_RAYLEIGH_MODES = 5    # Rayleigh modes for HV calculation
N_LOVE_MODES = 5        # Love modes for HV calculation
N_WAVENUMBERS = 1000      # integration points for body waves (0 = skip body waves)
APSV = 1e-3             # imaginary part of freq. to stabilize PSV (body waves)

# ---- Inversion settings ----
RANDOM_SEED = 42        # set to None for non-reproducible runs

# Stage 1: initial random search
N_RANDOM = 500          # number of random models in initial search

# Stage 2: choose method — 'montecarlo' or 'sa' (simulated annealing)
INVERSION_METHOD = "montecarlo"

# Monte Carlo / SA shared
N_ITERATIONS = 2000     # iterations per stage
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

# ---- Output ----
SAVE_FIGURE = "inversion_results.png"   # set to None to skip saving
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
        Percentage of measured value used as std when the file has no std column.

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


def make_forward_function(hvf_exe, fwd_cfg):
    """
    Return a forward function: f(model, obs) -> (hv_syn, dc_syn).
    hv_syn or dc_syn is None if that curve type is not requested.
    """
    def forward(model, obs):
        hv_syn = None
        dc_syn = None

        if obs.has_hv:
            hv_syn = compute_hv(
                model, obs.hv_freq, hvf_exe,
                n_rayleigh=fwd_cfg["n_rayleigh"],
                n_love=fwd_cfg["n_love"],
                n_wavenumbers=fwd_cfg["n_wavenumbers"],
                apsv=fwd_cfg["apsv"],
            )

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
        obs.hv_freq = hv_freq
        obs.hv_amp = hv_amp
        obs.hv_std = hv_std
        print(f"Loaded HV data: {len(hv_freq)} points  "
              f"[{hv_freq[0]:.3f} – {hv_freq[-1]:.3f} Hz]")

    if DC_FILE:
        dc_path = str(base / DC_FILE)
        dc_freq, dc_vel, dc_std = load_data(dc_path, DEFAULT_STD_PCT)
        dc_freq, dc_vel, dc_std = 1.0 / dc_freq, dc_vel * 1000.0, dc_std * 1000.0
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

    # ------------------------------------------------------------------
    # 3. Forward function
    # ------------------------------------------------------------------
    fwd_cfg = dict(
        n_rayleigh=N_RAYLEIGH_MODES,
        n_love=N_LOVE_MODES,
        n_wavenumbers=N_WAVENUMBERS,
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
            print(f"  [{n_eval:5d} models]  Best misfit = {best_misfit:.5f}")

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
    print_summary(final_result, obs)

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
        save_path=str(base / SAVE_FIGURE) if SAVE_FIGURE else None,
    )
    plt.show()


if __name__ == "__main__":
    main()
