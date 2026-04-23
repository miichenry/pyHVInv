"""
Inversion algorithms for HV-DC joint inversion.

Implements:
  - Random search (initial exploration)
  - Monte Carlo sampling (Metropolis-Hastings)
  - Simulated Annealing (SA)

Misfit (L2 norm):
  - Joint HV+DC: L2 = 2*(1-w)*sum((HV_syn-HV_obs)²/σ_HV²)
                      + 2*w*sum((DC_syn-DC_obs)²/σ_DC²)
  - HV only:     L2 = sum((HV_syn-HV_obs)²/σ_HV²)
  - DC only:     L2 = sum((DC_syn-DC_obs)²/σ_DC²)

where w = n_HV/(n_HV+n_DC) when equalize_weights=True.

References
----------
Garcia-Jerez et al. (2016) Computers & Geosciences.
Piña-Flores et al. (2017) Geophysical Journal International.
"""

import numpy as np
import time
from typing import Optional, Callable, Tuple
from dataclasses import dataclass, field


@dataclass
class ObservedCurves:
    """Container for observed data used in misfit calculation."""
    # HV curve
    hv_freq: Optional[np.ndarray] = None   # frequencies (Hz)
    hv_amp: Optional[np.ndarray] = None    # amplitudes
    hv_std: Optional[np.ndarray] = None    # standard deviations

    # Dispersion curve
    dc_freq: Optional[np.ndarray] = None   # frequencies (Hz)
    dc_vel: Optional[np.ndarray] = None    # phase velocities (m/s)
    dc_std: Optional[np.ndarray] = None    # standard deviations

    # DC curve type
    dc_polarization: str = "rayleigh"      # 'rayleigh' or 'love'
    dc_velocity_type: str = "phase"        # 'phase' or 'group'
    dc_mode: int = 0                       # mode number

    @property
    def has_hv(self) -> bool:
        return self.hv_freq is not None

    @property
    def has_dc(self) -> bool:
        return self.dc_freq is not None

    @property
    def n_hv(self) -> int:
        return len(self.hv_freq) if self.has_hv else 0

    @property
    def n_dc(self) -> int:
        return len(self.dc_freq) if self.has_dc else 0

    def dc_weight(self, equalize: bool = True, user_weight: float = 0.5) -> float:
        """Weight for DC misfit in joint inversion."""
        if equalize and self.has_hv and self.has_dc:
            return self.n_hv / (self.n_hv + self.n_dc)
        return user_weight


@dataclass
class InversionResult:
    """Results from the inversion."""
    best_model: np.ndarray          # best-fit model (nlayers, 4)
    best_misfit: float              # minimum misfit
    best_hv: Optional[np.ndarray]  # synthetic HV at best model
    best_dc: Optional[np.ndarray]  # synthetic DC at best model

    # History (for all accepted/evaluated models)
    all_models: list = field(default_factory=list)    # list of (nlayers,4) arrays
    all_misfits: list = field(default_factory=list)   # list of floats
    all_hv: list = field(default_factory=list)        # list of HV arrays
    all_dc: list = field(default_factory=list)        # list of DC arrays
    misfit_history: list = field(default_factory=list)  # best misfit per iteration

    n_evaluated: int = 0
    n_valid: int = 0
    elapsed_time: float = 0.0


def compute_misfit(
    hv_syn: Optional[np.ndarray],
    dc_syn: Optional[np.ndarray],
    obs: ObservedCurves,
    equalize: bool = True,
    user_dc_weight: float = 0.5,
) -> float:
    """
    Compute L2 misfit between synthetic and observed curves.

    Parameters
    ----------
    hv_syn : ndarray or None
        Synthetic H/V amplitudes.
    dc_syn : ndarray or None
        Synthetic dispersion velocities (m/s).
    obs : ObservedCurves
    equalize : bool
        Auto-equalize weights by number of samples.
    user_dc_weight : float
        Manual weight for DC (used if equalize=False).

    Returns
    -------
    misfit : float
    """
    has_hv = obs.has_hv and hv_syn is not None
    has_dc = obs.has_dc and dc_syn is not None

    if not has_hv and not has_dc:
        return np.inf

    if has_hv and has_dc:
        w = obs.dc_weight(equalize, user_dc_weight)
        hv_term = np.sum(((hv_syn - obs.hv_amp) / obs.hv_std) ** 2)
        dc_term = np.sum(((dc_syn - obs.dc_vel) / obs.dc_std) ** 2)
        return 2 * (1 - w) * hv_term + 2 * w * dc_term

    elif has_hv:
        return float(np.sum(((hv_syn - obs.hv_amp) / obs.hv_std) ** 2))

    else:
        return float(np.sum(((dc_syn - obs.dc_vel) / obs.dc_std) ** 2))


def run_random_search(
    var: np.ndarray,
    obs: ObservedCurves,
    forward_fn: Callable,
    n_models: int = 500,
    allow_lvz_vs: bool = True,
    allow_lvz_vp: bool = True,
    equalize_weights: bool = True,
    dc_weight: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    callback: Optional[Callable] = None,
) -> InversionResult:
    """
    Initial random search of parameter space.

    Parameters
    ----------
    var : ndarray (n_layers, 15)
        Parameter bounds.
    obs : ObservedCurves
        Observed data.
    forward_fn : callable
        Function(model, obs) -> (hv_syn, dc_syn). Returns None for invalid.
    n_models : int
        Number of random models to evaluate.
    allow_lvz_vs, allow_lvz_vp : bool
    equalize_weights, dc_weight : see compute_misfit
    rng : numpy random Generator
    callback : callable(iteration, best_misfit), optional
        Called after each evaluation.

    Returns
    -------
    result : InversionResult
    """
    from .model import random_model

    if rng is None:
        rng = np.random.default_rng()

    result = InversionResult(
        best_model=np.zeros((var.shape[0], 4)),
        best_misfit=np.inf,
        best_hv=None,
        best_dc=None,
    )

    t0 = time.time()

    for i in range(n_models):
        model = random_model(var, allow_lvz_vs, allow_lvz_vp, rng)
        if model is None:
            continue

        hv_syn, dc_syn = forward_fn(model, obs)

        misfit = compute_misfit(
            hv_syn, dc_syn, obs, equalize_weights, dc_weight
        )

        result.n_evaluated += 1
        if np.isfinite(misfit) and misfit > 0:
            result.n_valid += 1
            result.all_models.append(model.copy())
            result.all_misfits.append(misfit)
            if obs.has_hv and hv_syn is not None:
                result.all_hv.append(hv_syn.copy())
            if obs.has_dc and dc_syn is not None:
                result.all_dc.append(dc_syn.copy())

            if misfit < result.best_misfit:
                result.best_misfit = misfit
                result.best_model = model.copy()
                result.best_hv = hv_syn.copy() if hv_syn is not None else None
                result.best_dc = dc_syn.copy() if dc_syn is not None else None

        result.misfit_history.append(result.best_misfit)

        if callback:
            callback(i + 1, result.best_misfit)

    result.elapsed_time = time.time() - t0
    return result


def run_monte_carlo(
    initial_result: InversionResult,
    var: np.ndarray,
    obs: ObservedCurves,
    forward_fn: Callable,
    n_iterations: int = 1000,
    perturbation_pct: float = 5.0,
    allow_lvz_vs: bool = True,
    allow_lvz_vp: bool = True,
    equalize_weights: bool = True,
    dc_weight: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    callback: Optional[Callable] = None,
) -> InversionResult:
    """
    Monte Carlo sampling (Metropolis-Hastings) starting from initial_result.

    Acceptance rule: always accept if misfit decreases; accept with
    probability exp(-0.5 * delta_E) if misfit increases.

    Parameters
    ----------
    initial_result : InversionResult
        Result from random search (provides starting model).
    var : ndarray (n_layers, 15)
        Parameter bounds.
    obs : ObservedCurves
    forward_fn : callable
    n_iterations : int
    perturbation_pct : float
        Perturbation range as % of each parameter's range.
    allow_lvz_vs, allow_lvz_vp : bool
    equalize_weights, dc_weight : see compute_misfit
    rng : numpy random Generator
    callback : callable(iteration, best_misfit), optional

    Returns
    -------
    result : InversionResult (extends initial_result history)
    """
    from .model import perturb_model

    if rng is None:
        rng = np.random.default_rng()

    result = InversionResult(
        best_model=initial_result.best_model.copy(),
        best_misfit=initial_result.best_misfit,
        best_hv=initial_result.best_hv.copy() if initial_result.best_hv is not None else None,
        best_dc=initial_result.best_dc.copy() if initial_result.best_dc is not None else None,
        all_models=list(initial_result.all_models),
        all_misfits=list(initial_result.all_misfits),
        all_hv=list(initial_result.all_hv),
        all_dc=list(initial_result.all_dc),
        misfit_history=list(initial_result.misfit_history),
        n_evaluated=initial_result.n_evaluated,
        n_valid=initial_result.n_valid,
    )

    current_model = result.best_model.copy()
    current_misfit = result.best_misfit

    t0 = time.time()

    for i in range(n_iterations):
        candidate = perturb_model(
            current_model, var, perturbation_pct,
            allow_lvz_vs, allow_lvz_vp, rng
        )
        if candidate is None:
            continue

        hv_syn, dc_syn = forward_fn(candidate, obs)
        misfit = compute_misfit(hv_syn, dc_syn, obs, equalize_weights, dc_weight)

        result.n_evaluated += 1

        if np.isfinite(misfit) and misfit > 0:
            result.n_valid += 1
            delta_e = misfit - current_misfit

            # Metropolis acceptance
            accept = delta_e < 0 or rng.uniform() < np.exp(-0.5 * delta_e)

            if accept:
                current_model = candidate.copy()
                current_misfit = misfit

            # Always store all valid models for uncertainty analysis
            result.all_models.append(candidate.copy())
            result.all_misfits.append(misfit)
            if obs.has_hv and hv_syn is not None:
                result.all_hv.append(hv_syn.copy())
            if obs.has_dc and dc_syn is not None:
                result.all_dc.append(dc_syn.copy())

            if misfit < result.best_misfit:
                result.best_misfit = misfit
                result.best_model = candidate.copy()
                result.best_hv = hv_syn.copy() if hv_syn is not None else None
                result.best_dc = dc_syn.copy() if dc_syn is not None else None

        result.misfit_history.append(result.best_misfit)

        if callback:
            callback(result.n_evaluated, result.best_misfit)

    result.elapsed_time = time.time() - t0
    return result


def run_simulated_annealing(
    initial_result: InversionResult,
    var: np.ndarray,
    obs: ObservedCurves,
    forward_fn: Callable,
    n_iterations: int = 100,
    n_temperatures: int = 10,
    n_final_iter: int = 500,
    perturbation_pct: float = 5.0,
    initial_temperature: Optional[float] = None,
    cooling_rate: float = 0.95,
    allow_lvz_vs: bool = True,
    allow_lvz_vp: bool = True,
    equalize_weights: bool = True,
    dc_weight: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    callback: Optional[Callable] = None,
) -> InversionResult:
    """
    Simulated Annealing starting from initial_result.

    Schedule: n_temperatures Markov chains of n_iterations steps each,
    followed by a final chain of n_final_iter steps from the best model found.

    Parameters
    ----------
    initial_result : InversionResult
    var : ndarray (n_layers, 15)
    obs : ObservedCurves
    forward_fn : callable
    n_iterations : int
        Length of each Markov chain (per temperature step).
    n_temperatures : int
        Number of temperature reduction steps.
    n_final_iter : int
        Length of final chain (run at lowest temperature from best model).
    perturbation_pct : float
    initial_temperature : float, optional
        Starting temperature. If None, estimated from initial misfit.
    cooling_rate : float
        Multiplicative cooling factor (0 < rate < 1).
    allow_lvz_vs, allow_lvz_vp : bool
    equalize_weights, dc_weight : see compute_misfit
    rng : numpy random Generator
    callback : callable(iteration, best_misfit, temperature), optional

    Returns
    -------
    result : InversionResult
    """
    from .model import perturb_model

    if rng is None:
        rng = np.random.default_rng()

    result = InversionResult(
        best_model=initial_result.best_model.copy(),
        best_misfit=initial_result.best_misfit,
        best_hv=initial_result.best_hv.copy() if initial_result.best_hv is not None else None,
        best_dc=initial_result.best_dc.copy() if initial_result.best_dc is not None else None,
        all_models=list(initial_result.all_models),
        all_misfits=list(initial_result.all_misfits),
        all_hv=list(initial_result.all_hv),
        all_dc=list(initial_result.all_dc),
        misfit_history=list(initial_result.misfit_history),
        n_evaluated=initial_result.n_evaluated,
        n_valid=initial_result.n_valid,
    )

    # Estimate initial temperature from acceptance probability (~0.3) if not given
    if initial_temperature is None:
        # T such that exp(-L2_best / T) ~ 0.3
        if np.isfinite(result.best_misfit) and result.best_misfit > 0:
            initial_temperature = -result.best_misfit / np.log(0.3)
        else:
            initial_temperature = 1.0

    T = initial_temperature
    current_model = result.best_model.copy()
    current_misfit = result.best_misfit

    t0 = time.time()

    # Main SA loop
    for t_step in range(n_temperatures + 1):
        is_final = (t_step == n_temperatures)

        if is_final:
            # Final chain: start from best model at lowest temperature
            current_model = result.best_model.copy()
            current_misfit = result.best_misfit
            n_iter = n_final_iter
        else:
            n_iter = n_iterations

        for i in range(n_iter):
            candidate = perturb_model(
                current_model, var, perturbation_pct,
                allow_lvz_vs, allow_lvz_vp, rng
            )
            if candidate is None:
                continue

            hv_syn, dc_syn = forward_fn(candidate, obs)
            misfit = compute_misfit(hv_syn, dc_syn, obs, equalize_weights, dc_weight)

            result.n_evaluated += 1

            if np.isfinite(misfit) and misfit > 0:
                result.n_valid += 1
                delta_e = misfit - current_misfit

                # Boltzmann acceptance
                if delta_e < 0:
                    accept = True
                else:
                    accept = rng.uniform() < np.exp(-delta_e / T)

                if accept:
                    current_model = candidate.copy()
                    current_misfit = misfit

                # Store all valid models
                result.all_models.append(candidate.copy())
                result.all_misfits.append(misfit)
                if obs.has_hv and hv_syn is not None:
                    result.all_hv.append(hv_syn.copy())
                if obs.has_dc and dc_syn is not None:
                    result.all_dc.append(dc_syn.copy())

                if misfit < result.best_misfit:
                    result.best_misfit = misfit
                    result.best_model = candidate.copy()
                    result.best_hv = hv_syn.copy() if hv_syn is not None else None
                    result.best_dc = dc_syn.copy() if dc_syn is not None else None

            result.misfit_history.append(result.best_misfit)

            if callback:
                callback(result.n_evaluated, result.best_misfit, T)

        # Cool down
        if not is_final:
            T *= cooling_rate

    result.elapsed_time = time.time() - t0
    return result
