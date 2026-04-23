"""
Earth model parameterization and random model generation.

The model parameter space is defined by a bounds matrix (shape nlayers x 15):
    cols 0-2:   thickness min, max, amp   (m)
    cols 3-5:   Vp min, max, amp          (m/s)
    cols 6-8:   Vs min, max, amp          (m/s)
    cols 9-11:  density min, max, amp     (kg/m³)
    cols 12-14: Poisson ratio min, max, amp

The last layer is the halfspace; its thickness is always 0.

References
----------
Garcia-Jerez et al. (2016) Computers & Geosciences.
Piña-Flores et al. (2017) Geophysical Journal International.
"""

import numpy as np
from typing import Optional


def make_bounds(
    n_layers: int,
    h_min: list, h_max: list,
    vp_min: list, vp_max: list,
    vs_min: list, vs_max: list,
    rho_min: list, rho_max: list,
    nu_min: Optional[list] = None,
    nu_max: Optional[list] = None,
) -> np.ndarray:
    """
    Construct the bounds matrix for the inversion.

    Parameters
    ----------
    n_layers : int
        Number of layers including halfspace.
    h_min, h_max : list of float, length n_layers
        Thickness bounds (m). Last layer (halfspace) should be [0, 0].
    vp_min, vp_max : list of float
        P-wave velocity bounds (m/s).
    vs_min, vs_max : list of float
        S-wave velocity bounds (m/s).
    rho_min, rho_max : list of float
        Density bounds (kg/m³).
    nu_min, nu_max : list of float, optional
        Poisson's ratio bounds. Defaults to [0.1, 0.49] for all layers.

    Returns
    -------
    var : ndarray, shape (n_layers, 15)
    """
    if nu_min is None:
        nu_min = [0.1] * n_layers
    if nu_max is None:
        nu_max = [0.49] * n_layers

    var = np.zeros((n_layers, 15))
    for i in range(n_layers):
        var[i, 0] = h_min[i]
        var[i, 1] = h_max[i]
        var[i, 2] = h_max[i] - h_min[i]
        var[i, 3] = vp_min[i]
        var[i, 4] = vp_max[i]
        var[i, 5] = vp_max[i] - vp_min[i]
        var[i, 6] = vs_min[i]
        var[i, 7] = vs_max[i]
        var[i, 8] = vs_max[i] - vs_min[i]
        var[i, 9] = rho_min[i]
        var[i, 10] = rho_max[i]
        var[i, 11] = rho_max[i] - rho_min[i]
        var[i, 12] = nu_min[i]
        var[i, 13] = nu_max[i]
        var[i, 14] = nu_max[i] - nu_min[i]

    return var


def validate_and_fix_bounds(
    var: np.ndarray,
    allow_lvz_vs: bool = True,
    allow_lvz_vp: bool = True,
) -> np.ndarray:
    """
    Check and fix parameter bounds compatibility, matching the HV-INV ini2A.m logic.

    Three constraints are enforced iteratively until all pass:

    1. Poisson compatibility — when LVZ in Vp is allowed (primary variable = Vs):
           Vs_min = max(Vs_min, Vp_min * sqrt((1-2*nu_max) / (2*(1-nu_max))))
           Vs_max = min(Vs_max, Vp_max * sqrt((1-2*nu_min) / (2*(1-nu_min))))
       When LVZ in Vs is allowed (primary variable = Vp):
           Vp_min = max(Vp_min, Vs_min * sqrt(2*(1-nu_min) / (1-2*nu_min)))
           Vp_max = min(Vp_max, Vs_max * sqrt(2*(1-nu_max) / (1-2*nu_max)))

    2. Monotonicity propagation (when LVZ is forbidden) — bounds of adjacent
       layers are made mutually consistent:
           upper bound of layer k <= upper bound of layer k+1
           lower bound of layer k+1 >= lower bound of layer k

    3. Collapsed interval handling — if after clipping the interval collapses
       (min >= max), both bounds are pinned to their midpoint so the layer
       is effectively fixed.

    The loop repeats until all three constraints are simultaneously satisfied,
    matching the `while ~all_correct` loop in ini2A.m.

    Parameters
    ----------
    var : ndarray, shape (n_layers, 15)
        Bounds matrix from make_bounds().
    allow_lvz_vs : bool
        If False, Vs must increase with depth (monotonicity enforced).
    allow_lvz_vp : bool
        If False, Vp must increase with depth (monotonicity enforced).

    Returns
    -------
    var : ndarray
        Corrected bounds matrix.
    """
    var = var.copy()
    n_layers = var.shape[0]
    epsil_vel = 1.0   # collapsed interval threshold (m/s), matches MATLAB epsil=1e0
    epsil_check = 1e-3  # convergence threshold, matches MATLAB epsil=1e-3

    varoriginal = var.copy()
    not_collapsed = np.ones(n_layers, dtype=bool)

    for iteration in range(200):  # safety cap
        all_correct = True

        # ------------------------------------------------------------------
        # 1. Poisson compatibility
        # ------------------------------------------------------------------
        if allow_lvz_vp:
            # Primary variable is Vs; clip Vs to be compatible with Vp + Poisson
            nu_min = var[:, 12]
            nu_max = var[:, 13]
            vs_min_compat = var[:, 3] * np.sqrt((1 - 2 * nu_max) / (2 * (1 - nu_max)))
            vs_max_compat = var[:, 5] * np.sqrt((1 - 2 * nu_min) / (2 * (1 - nu_min)))  # col 5 = Vp_max

            var[not_collapsed, 6] = np.maximum(var[not_collapsed, 6], vs_min_compat[not_collapsed])
            var[not_collapsed, 7] = np.minimum(var[not_collapsed, 7], vs_max_compat[not_collapsed])
            var[not_collapsed, 8] = var[not_collapsed, 7] - var[not_collapsed, 6]

            # Collapsed interval: pin to midpoint
            collapsed = not_collapsed & (var[:, 8] < epsil_vel)
            if collapsed.any():
                mid = (var[collapsed, 6] + var[collapsed, 7]) / 2
                var[collapsed, 6] = mid
                var[collapsed, 7] = mid
                var[collapsed, 8] = 0.0
                not_collapsed[collapsed] = False

            # Convergence check: are all effective Vs ranges now consistent?
            effective_min = np.maximum(var[:, 6], vs_min_compat)
            effective_max = np.minimum(var[:, 7], vs_max_compat)
            if np.any(effective_min - effective_max > epsil_check):
                all_correct = False

        else:
            # Primary variable is Vp; clip Vp to be compatible with Vs + Poisson
            nu_min = var[:, 12]
            nu_max = var[:, 13]
            vp_min_compat = var[:, 6] * np.sqrt((2 * (1 - nu_min)) / (1 - 2 * nu_min))
            vp_max_compat = var[:, 7] * np.sqrt((2 * (1 - nu_max)) / (1 - 2 * nu_max))

            var[not_collapsed, 3] = np.maximum(var[not_collapsed, 3], vp_min_compat[not_collapsed])
            var[not_collapsed, 4] = np.minimum(var[not_collapsed, 4], vp_max_compat[not_collapsed])
            var[not_collapsed, 5] = var[not_collapsed, 4] - var[not_collapsed, 3]

            # Collapsed interval: pin to midpoint
            collapsed = not_collapsed & (var[:, 5] < epsil_vel)
            if collapsed.any():
                mid = (var[collapsed, 3] + var[collapsed, 4]) / 2
                var[collapsed, 3] = mid
                var[collapsed, 4] = mid
                var[collapsed, 5] = 0.0
                not_collapsed[collapsed] = False

            effective_min = np.maximum(var[:, 3], vp_min_compat)
            effective_max = np.minimum(var[:, 4], vp_max_compat)
            if np.any(effective_min - effective_max > epsil_check):
                all_correct = False

        # ------------------------------------------------------------------
        # 2. Monotonicity propagation (when LVZ forbidden)
        # ------------------------------------------------------------------
        if not allow_lvz_vs:
            # Upper Vs bound of layer k <= upper bound of layer k+1
            for k in range(n_layers - 2, -1, -1):
                var[k, 7] = min(var[k, 7], var[k + 1, 7])
            # Lower Vs bound of layer k >= lower bound of layer k-1
            for k in range(1, n_layers):
                var[k, 6] = max(var[k, 6], var[k - 1, 6])
            var[:, 8] = var[:, 7] - var[:, 6]

            if (np.any(np.diff(var[:, 6]) < 0) or np.any(np.diff(var[:, 7]) < 0)):
                all_correct = False

        if not allow_lvz_vp:
            # Same for Vp
            for k in range(n_layers - 2, -1, -1):
                var[k, 4] = min(var[k, 4], var[k + 1, 4])
            for k in range(1, n_layers):
                var[k, 3] = max(var[k, 3], var[k - 1, 3])
            var[:, 5] = var[:, 4] - var[:, 3]

            if (np.any(np.diff(var[:, 3]) < 0) or np.any(np.diff(var[:, 4]) < 0)):
                all_correct = False

        if all_correct:
            break

    # ------------------------------------------------------------------
    # Report changes
    # ------------------------------------------------------------------
    changed_cols = np.any(var != varoriginal, axis=0)
    any_issue = changed_cols.any()

    if any_issue:
        layer_labels = [f"layer {k+1}" if k < n_layers - 1 else "halfspace"
                        for k in range(n_layers)]
        for k in range(n_layers):
            msgs = []
            if var[k, 6] != varoriginal[k, 6] or var[k, 7] != varoriginal[k, 7]:
                msgs.append(
                    f"  Vs: [{varoriginal[k,6]:.1f}, {varoriginal[k,7]:.1f}] "
                    f"→ [{var[k,6]:.1f}, {var[k,7]:.1f}] m/s"
                )
            if var[k, 3] != varoriginal[k, 3] or var[k, 4] != varoriginal[k, 4]:
                msgs.append(
                    f"  Vp: [{varoriginal[k,3]:.1f}, {varoriginal[k,4]:.1f}] "
                    f"→ [{var[k,3]:.1f}, {var[k,4]:.1f}] m/s"
                )
            if msgs:
                print(f"WARNING ({layer_labels[k]}): bounds updated to satisfy constraints:")
                for m in msgs:
                    print(m)
        print("Some values in intervals were incompatible with other constraints — ranges updated.")
    else:
        print("Bounds check passed: all ranges are compatible.")

    return var


def _vs_from_vp_poisson(vp: float, nu: float) -> float:
    """Compute Vs from Vp and Poisson's ratio."""
    return vp * np.sqrt((1 - 2 * nu) / (2 * (1 - nu)))


def _vp_from_vs_poisson(vs: float, nu: float) -> float:
    """Compute Vp from Vs and Poisson's ratio."""
    return vs * np.sqrt(2 * (1 - nu) / (1 - 2 * nu))


def random_model(
    var: np.ndarray,
    allow_lvz_vs: bool = True,
    allow_lvz_vp: bool = True,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 1000,
) -> Optional[np.ndarray]:
    """
    Generate a random earth model within the given parameter bounds.

    Parameters
    ----------
    var : ndarray, shape (n_layers, 15)
        Parameter bounds as returned by make_bounds().
    allow_lvz_vs : bool
        If False, Vs must increase monotonically with depth.
    allow_lvz_vp : bool
        If False, Vp must increase monotonically with depth.
    rng : numpy random Generator, optional
    max_attempts : int
        Max rejection-sampling attempts before giving up.

    Returns
    -------
    model : ndarray, shape (n_layers, 4) or None
        Columns: [thickness(m), Vp(m/s), Vs(m/s), density(kg/m³)].
        Last row has thickness=0 (halfspace).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_layers = var.shape[0]

    for _ in range(max_attempts):
        model = np.zeros((n_layers, 4))

        # --- Thickness (last layer = halfspace, thickness = 0) ---
        for k in range(n_layers - 1):
            amp = var[k, 2]
            if amp > 0:
                model[k, 0] = var[k, 0] + amp * rng.uniform()
            else:
                model[k, 0] = var[k, 0]
        model[-1, 0] = 0.0

        # --- Generate Vs (primary velocity) ---
        if not allow_lvz_vs:
            # Must be monotonically increasing: generate in order
            vs_prev = 0.0
            valid_vs = True
            for k in range(n_layers):
                lo = max(var[k, 6], vs_prev)
                hi = var[k, 7]
                if lo > hi + 1e-3:
                    valid_vs = False
                    break
                hi = max(hi, lo)
                model[k, 2] = lo + (hi - lo) * rng.uniform()
                vs_prev = model[k, 2]
            if not valid_vs:
                continue
        else:
            # LVZ allowed: uniform random in range
            for k in range(n_layers):
                amp = var[k, 8]
                if amp > 0:
                    model[k, 2] = var[k, 6] + amp * rng.uniform()
                else:
                    model[k, 2] = var[k, 6]

        # --- Generate Vp consistent with Vs and Poisson bounds ---
        valid_vp = True
        for k in range(n_layers):
            vs = model[k, 2]
            nu_min = var[k, 12]
            nu_max = var[k, 13]

            # Vp range from Poisson ratio constraint
            vp_from_nu_min = _vp_from_vs_poisson(vs, nu_min)
            vp_from_nu_max = _vp_from_vs_poisson(vs, nu_max)
            vp_lo_poisson = min(vp_from_nu_min, vp_from_nu_max)
            vp_hi_poisson = max(vp_from_nu_min, vp_from_nu_max)

            # Intersect with user-specified Vp bounds
            vp_lo = max(var[k, 3], vp_lo_poisson)
            vp_hi = min(var[k, 4], vp_hi_poisson)

            if not allow_lvz_vp and k > 0:
                vp_lo = max(vp_lo, model[k - 1, 1])

            if vp_lo > vp_hi + 1e-3:
                valid_vp = False
                break

            vp_hi = max(vp_hi, vp_lo)
            model[k, 1] = vp_lo + (vp_hi - vp_lo) * rng.uniform()

        if not valid_vp:
            continue

        # --- Density ---
        for k in range(n_layers):
            amp = var[k, 11]
            if amp > 0:
                model[k, 3] = var[k, 9] + amp * rng.uniform()
            else:
                model[k, 3] = var[k, 9]

        return model

    return None  # Failed to generate valid model


def perturb_model(
    model: np.ndarray,
    var: np.ndarray,
    perturbation_pct: float,
    allow_lvz_vs: bool = True,
    allow_lvz_vp: bool = True,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 200,
) -> Optional[np.ndarray]:
    """
    Generate a perturbed model around the current best model.

    Perturbation is bounded by both the global parameter ranges (var) and
    a perturbation window of ±(perturbation_pct/100) * range around the
    current model values.

    Parameters
    ----------
    model : ndarray, shape (n_layers, 4)
        Current model to perturb.
    var : ndarray, shape (n_layers, 15)
        Global parameter bounds.
    perturbation_pct : float
        Maximum perturbation as percentage of each parameter's range.
    allow_lvz_vs, allow_lvz_vp : bool
        Allow low-velocity zones.
    rng : numpy random Generator, optional
    max_attempts : int

    Returns
    -------
    perturbed : ndarray, shape (n_layers, 4) or None
    """
    if rng is None:
        rng = np.random.default_rng()

    nn = perturbation_pct / 100.0
    n_layers = var.shape[0]

    # Build tighter bounds: intersection of global range and perturbation window
    var_local = var.copy()
    for col_idx, amp_col in enumerate([2, 5, 8, 11]):  # h, vp, vs, rho
        param_col = col_idx  # model column index
        min_col = amp_col - 2
        max_col = amp_col - 1

        step = var[:, amp_col] * nn
        var_local[:, min_col] = np.maximum(var[:, min_col], model[:, param_col] - step)
        var_local[:, max_col] = np.minimum(var[:, max_col], model[:, param_col] + step)
        var_local[:, amp_col] = var_local[:, max_col] - var_local[:, min_col]

    return random_model(var_local, allow_lvz_vs, allow_lvz_vp, rng, max_attempts)


def model_to_depth_profile(model: np.ndarray, max_depth: Optional[float] = None):
    """
    Convert layer model (thickness) to a step-function depth profile for plotting.

    Each layer produces two depth points (top and bottom at the same velocity),
    so that a standard line plot renders as a staircase.

    Returns
    -------
    depth_plot : ndarray, shape (2*n_layers,)
    vs_plot    : ndarray, shape (2*n_layers,)
    vp_plot    : ndarray, shape (2*n_layers,)
    """
    n = model.shape[0]
    interfaces = np.cumsum(model[:, 0])  # cumulative depths of layer bottoms

    if max_depth is None:
        max_depth = interfaces[-2] * 1.5 if n >= 2 else 100.0

    depth_plot = np.empty(2 * n)
    vs_plot = np.empty(2 * n)
    vp_plot = np.empty(2 * n)

    for k in range(n):
        top = interfaces[k - 1] if k > 0 else 0.0
        bottom = interfaces[k] if k < n - 1 else max_depth

        depth_plot[2 * k] = top
        depth_plot[2 * k + 1] = bottom
        vs_plot[2 * k] = model[k, 2]
        vs_plot[2 * k + 1] = model[k, 2]
        vp_plot[2 * k] = model[k, 1]
        vp_plot[2 * k + 1] = model[k, 1]

    return depth_plot, vs_plot, vp_plot
