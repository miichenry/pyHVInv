"""
Forward model wrapper for HVf executable.

HVf computes H/V spectral ratios and dispersion curves for a layered earth
model using the Diffuse Field Assumption (Sanchez-Sesma et al. 2011).

Model format (rows = layers, last row = halfspace with thickness=0):
    col 0: thickness (m)
    col 1: Vp (m/s)
    col 2: Vs (m/s)
    col 3: density (kg/m³)
"""

import subprocess
import tempfile
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def _write_model_file(model: np.ndarray, path: str) -> None:
    """Write layered model to file in HVf format."""
    n_layers = model.shape[0]
    with open(path, "w") as f:
        f.write(f"{n_layers}\n")
        for row in model:
            f.write(f"{row[0]:.4f}\t{row[1]:.4f}\t{row[2]:.4f}\t{row[3]:.4f}\n")


def _write_freq_file(frequencies: np.ndarray, path: str) -> None:
    """Write frequency list to file."""
    np.savetxt(path, frequencies, fmt="%.8f")


def compute_hv(
    model: np.ndarray,
    frequencies: np.ndarray,
    hvf_exe: str,
    n_rayleigh: int = 3,
    n_love: int = 3,
    n_wavenumbers: int = 50,
    apsv: float = 1e-4,
    ash: float = 0.01,
) -> Optional[np.ndarray]:
    """
    Compute H/V spectral ratio for a layered model.

    Parameters
    ----------
    model : ndarray, shape (nlayers, 4)
        Layered earth model [thickness, Vp, Vs, density].
        Last row is halfspace (thickness=0).
    frequencies : ndarray
        Frequency vector (Hz).
    hvf_exe : str
        Path to HVf executable.
    n_rayleigh : int
        Max number of Rayleigh modes.
    n_love : int
        Max number of Love modes.
    n_wavenumbers : int
        Integration points for body-wave integrals (0 to skip body waves).
    apsv : float
        Imaginary frequency part to stabilize PSV integrals.
    ash : float
        Imaginary frequency part to stabilize SH integrals.

    Returns
    -------
    hv_amp : ndarray or None
        H/V amplitude at each frequency. None if calculation failed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = os.path.join(tmpdir, "model.txt")
        freq_file = os.path.join(tmpdir, "freq.txt")

        _write_model_file(model, model_file)
        _write_freq_file(frequencies, freq_file)

        cmd = [
            hvf_exe,
            "-nmr", str(n_rayleigh),
            "-nml", str(n_love),
            "-nks", str(n_wavenumbers),
            "-apsv", str(apsv),
            "-ash", str(ash),
            "-hv",
            "-ff", freq_file,
            "-f", model_file,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            output = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        if not output:
            return None

        # Output format: pairs (freq, amplitude) on one line
        try:
            values = np.array([float(x) for x in output.split()])
        except ValueError:
            return None

        if len(values) < 2:
            return None

        values = values.reshape(-1, 2)
        hv_amp = values[:, 1]

        if len(hv_amp) != len(frequencies):
            return None
        if np.any(np.isnan(hv_amp)) or np.any(np.isinf(hv_amp)) or np.any(hv_amp == 0):
            return None

        return hv_amp


def compute_dc(
    model: np.ndarray,
    frequencies: np.ndarray,
    hvf_exe: str,
    polarization: str = "rayleigh",
    velocity_type: str = "phase",
    mode: int = 0,
) -> Optional[np.ndarray]:
    """
    Compute dispersion curve (phase or group velocity) for a layered model.

    Parameters
    ----------
    model : ndarray, shape (nlayers, 4)
        Layered earth model [thickness, Vp, Vs, density].
        Last row is halfspace (thickness=0).
    frequencies : ndarray
        Frequency vector (Hz).
    hvf_exe : str
        Path to HVf executable.
    polarization : str
        'rayleigh' or 'love'.
    velocity_type : str
        'phase' or 'group'.
    mode : int
        Mode number (0 = fundamental).

    Returns
    -------
    velocities : ndarray or None
        Phase/group velocities (m/s) at each frequency. None if failed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = os.path.join(tmpdir, "model.txt")
        freq_file = os.path.join(tmpdir, "freq.txt")

        _write_model_file(model, model_file)
        _write_freq_file(frequencies, freq_file)

        n_modes = mode + 1  # HVf expects total number of modes to compute
        vel_flag = "-ph" if velocity_type == "phase" else "-gr"

        if polarization == "rayleigh":
            cmd = [hvf_exe, "-nmr", str(n_modes), "-nml", "0",
                   "-prec", "1e-10", vel_flag, "-ff", freq_file, "-f", model_file]
        else:
            cmd = [hvf_exe, "-nml", str(n_modes), "-nmr", "0",
                   "-prec", "1e-10", vel_flag, "-ff", freq_file, "-f", model_file]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            output = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        if not output:
            return None

        # Output format: "N_freq N_modes\n  s1 s2 ... sN*NM\n  T T T..."
        # Parse all floats (T/F logical values are skipped by float conversion)
        try:
            lines = output.split("\n")
            all_values = []
            for line in lines:
                for token in line.split():
                    try:
                        all_values.append(float(token))
                    except ValueError:
                        pass  # skip 'T', 'F', etc.
        except Exception:
            return None

        if len(all_values) < 3:
            return None

        # First two values: n_freq, n_modes_computed
        n_freq = int(all_values[0])
        n_modes_out = int(all_values[1])
        slowness = np.array(all_values[2:])

        # The fundamental mode corresponds to the first n_freq values
        if len(slowness) < n_freq:
            return None

        # Extract the requested mode
        # Layout: mode_0 at f1..fN, mode_1 at f1..fN, ...
        mode_idx = min(mode, n_modes_out - 1)
        slow_mode = slowness[mode_idx * n_freq: (mode_idx + 1) * n_freq]

        if len(slow_mode) != len(frequencies):
            return None

        with np.errstate(divide="ignore", invalid="ignore"):
            velocities = np.where(slow_mode > 0, 1.0 / slow_mode, np.nan)

        # Quality check: interpolate small gaps, reject if too many NaN/zeros
        bad = np.isnan(velocities) | np.isinf(velocities) | (velocities == 0)
        if bad.sum() > 0.1 * len(velocities):
            return None
        if bad.any():
            freq_ok = frequencies[~bad]
            vel_ok = velocities[~bad]
            velocities[bad] = np.interp(frequencies[bad], freq_ok, vel_ok)

        return velocities
