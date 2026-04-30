"""
PatchBatch Electrophysiology Data Analysis Tool - Kinetics Service

Stateless service for fitting mono- and bi-exponential models to current traces
within a user-defined time window. Auto-detects rising vs. decaying direction and
reports per-fit parameters, their standard errors, goodness-of-fit statistics,
and AIC/BIC for model comparison.

The fitting form is selected based on detected direction:
    Rising:   I(t) = A * (1 - exp(-t / tau)) + C            (mono)
              I(t) = A1*(1 - exp(-t/tau1)) + A2*(1 - exp(-t/tau2)) + C   (bi)
    Decaying: I(t) = A * exp(-t / tau) + C                   (mono)
              I(t) = A1*exp(-t/tau1) + A2*exp(-t/tau2) + C   (bi)

Time origin t=0 corresponds to the start of the fit region (either range_start_ms
or the auto-detected sub-region start).

An optional auto_detect_region flag narrows the fit to a sub-region anchored on
the peak (decaying) or trough (rising) within the user-specified range. This is
peak-anchored detection: for a decaying trace, fitting begins at the maximum
sample and extends to the end of the range; for a rising trace, fitting begins
at the minimum sample and extends to the end of the range.

This module has no Qt dependencies per the project's layered architecture.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


# Smoothing window (in samples) applied before peak/trough detection in
# auto-detect mode. Kept very small per design choice: manual range selection
# is expected to exclude noise spikes, so only minimal smoothing is needed to
# avoid single-sample jitter from defining the anchor.
_AUTO_DETECT_SMOOTH_WINDOW = 3


# --- Model functions ---

def _mono_rising(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    return A * (1.0 - np.exp(-t / tau)) + C


def _mono_decaying(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    return A * np.exp(-t / tau) + C


def _biexp_rising(t: np.ndarray, A1: float, tau1: float,
                  A2: float, tau2: float, C: float) -> np.ndarray:
    return A1 * (1.0 - np.exp(-t / tau1)) + A2 * (1.0 - np.exp(-t / tau2)) + C


def _biexp_decaying(t: np.ndarray, A1: float, tau1: float,
                    A2: float, tau2: float, C: float) -> np.ndarray:
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C


# --- Result containers ---

@dataclass(frozen=True)
class SingleFitResult:
    """Result of a single model fit (mono- or bi-exponential)."""
    success: bool
    model_name: str
    error_message: Optional[str] = None
    params: Dict[str, float] = field(default_factory=dict)
    param_stderr: Dict[str, float] = field(default_factory=dict)
    r_squared: float = float("nan")
    adjusted_r_squared: float = float("nan")
    rmse: float = float("nan")
    ss_res: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    n_points: int = 0
    n_params: int = 0


@dataclass(frozen=True)
class KineticsResult:
    """Full kinetics analysis result: both fits plus comparison metrics.

    Fields:
        range_start_ms / range_end_ms: The user-specified range (Range 1) that
            was passed in. Preserved for reference/display.
        fit_region_start_ms / fit_region_end_ms: The actual time window used
            for fitting. Equals range_start_ms / range_end_ms when
            auto_detected is False. When True, the start may be later than
            range_start_ms because it was anchored on the detected
            peak/trough.
        auto_detected: Whether auto-detection was applied and actually
            narrowed the fit window.
    """
    success: bool
    error_message: Optional[str]
    direction: str                       # "rising" or "decaying"
    range_start_ms: float
    range_end_ms: float
    fit_region_start_ms: float
    fit_region_end_ms: float
    auto_detected: bool
    time_fit_ms: np.ndarray              # time axis used for fitting (t=0 at fit region start)
    current_fit: np.ndarray              # current samples used
    mono: Optional[SingleFitResult]
    biexp: Optional[SingleFitResult]
    delta_aic: Optional[float]           # biexp AIC - mono AIC (negative favors biexp)
    delta_bic: Optional[float]           # biexp BIC - mono BIC (negative favors biexp)


# --- Helpers ---

def _detect_direction(current: np.ndarray) -> str:
    """Compare mean of first and last ~10% of samples to detect rising vs decaying."""
    n = len(current)
    window = max(1, n // 10)
    start_mean = float(np.mean(current[:window]))
    end_mean = float(np.mean(current[-window:]))
    direction = "rising" if end_mean >= start_mean else "decaying"
    logger.debug(
        f"Direction detection: start_mean={start_mean:.4g}, end_mean={end_mean:.4g} "
        f"-> {direction}"
    )
    return direction


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """
    Simple centered moving average. Returns an array the same length as y.
    For window <= 1 or n <= window, returns y unchanged.
    """
    n = len(y)
    if window <= 1 or n <= window:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    # 'same' keeps length equal to n; edge samples are smoothed against zero-pad
    # equivalents, but for a window of 3 the edge distortion is negligible and
    # peak/trough detection is dominated by interior samples.
    return np.convolve(y, kernel, mode="same")


def _auto_detect_subregion(
    t: np.ndarray, y: np.ndarray, direction: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Narrow the fit region to the peak/trough-anchored sub-window.

    For decaying traces, the sub-region begins at the maximum of the (lightly
    smoothed) trace and extends to the last sample. For rising traces, it
    begins at the minimum.

    Returns (t_sub, y_sub, start_idx). If narrowing would leave fewer than 10
    samples, returns the original arrays with start_idx=0.
    """
    n = len(y)
    if n < 10:
        return t, y, 0

    smoothed = _moving_average(y, _AUTO_DETECT_SMOOTH_WINDOW)

    if direction == "decaying":
        start_idx = int(np.argmax(smoothed))
    else:
        start_idx = int(np.argmin(smoothed))

    # Require at least 10 samples post-anchor for a meaningful fit.
    if n - start_idx < 10:
        logger.warning(
            f"Auto-detect would leave only {n - start_idx} samples "
            f"(need >= 10); falling back to full user range."
        )
        return t, y, 0

    logger.debug(
        f"Auto-detect ({direction}): anchor at index {start_idx}/{n-1}, "
        f"t={t[start_idx]:.3f} ms, y={y[start_idx]:.4g}"
    )
    return t[start_idx:], y[start_idx:], start_idx


def _compute_fit_stats(
    y_obs: np.ndarray, y_pred: np.ndarray, n_params: int
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute R^2, adjusted R^2, RMSE, SS_res, AIC, BIC.

    AIC/BIC use the Gaussian-residuals form:
        AIC = n * ln(SS_res / n) + 2k
        BIC = n * ln(SS_res / n) + k * ln(n)
    Constants independent of the model are omitted, so these values are only
    meaningful as differences between models fit to the same data.
    """
    n = len(y_obs)
    residuals = y_obs - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Adjusted R^2 only valid when n - n_params - 1 > 0
    if n - n_params - 1 > 0 and not np.isnan(r_squared):
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - n_params - 1)
    else:
        adj_r_squared = float("nan")

    rmse = float(np.sqrt(ss_res / n)) if n > 0 else float("nan")

    # AIC/BIC (Gaussian form, constants dropped)
    if ss_res > 0 and n > 0:
        aic = n * np.log(ss_res / n) + 2 * n_params
        bic = n * np.log(ss_res / n) + n_params * np.log(n)
    else:
        aic = float("nan")
        bic = float("nan")

    return r_squared, adj_r_squared, rmse, ss_res, aic, bic


def _initial_guesses_mono(
    t: np.ndarray, y: np.ndarray, direction: str
) -> Tuple[list, list, list]:
    """Return (p0, lower_bounds, upper_bounds) for mono-exp fit."""
    y_span = float(np.max(y) - np.min(y))
    t_span = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    if direction == "rising":
        # End value ~ A + C; start value ~ C
        C_guess = float(np.mean(y[: max(1, len(y) // 20)]))
        A_guess = float(np.mean(y[-max(1, len(y) // 20):])) - C_guess
    else:
        # Start value ~ A + C; end value ~ C
        C_guess = float(np.mean(y[-max(1, len(y) // 20):]))
        A_guess = float(np.mean(y[: max(1, len(y) // 20)])) - C_guess

    if abs(A_guess) < 1e-12:
        A_guess = y_span if y_span > 0 else 1.0

    tau_guess = t_span / 3.0 if t_span > 0 else 1.0

    p0 = [A_guess, tau_guess, C_guess]
    lower = [-np.inf, 1e-9, -np.inf]
    upper = [np.inf, np.inf, np.inf]
    return p0, lower, upper


def _initial_guesses_biexp(
    t: np.ndarray, y: np.ndarray, direction: str
) -> Tuple[list, list, list]:
    """Return (p0, lower_bounds, upper_bounds) for bi-exp fit."""
    t_span = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    if direction == "rising":
        C_guess = float(np.mean(y[: max(1, len(y) // 20)]))
        total_A = float(np.mean(y[-max(1, len(y) // 20):])) - C_guess
    else:
        C_guess = float(np.mean(y[-max(1, len(y) // 20):]))
        total_A = float(np.mean(y[: max(1, len(y) // 20)])) - C_guess

    if abs(total_A) < 1e-12:
        total_A = float(np.max(y) - np.min(y)) or 1.0

    # Split amplitude 50/50; seed fast and slow taus at 20% and 80% of window
    A1_guess = total_A / 2.0
    A2_guess = total_A / 2.0
    tau1_guess = max(t_span * 0.2, 1e-6)
    tau2_guess = max(t_span * 0.8, tau1_guess * 2)

    p0 = [A1_guess, tau1_guess, A2_guess, tau2_guess, C_guess]
    lower = [-np.inf, 1e-9, -np.inf, 1e-9, -np.inf]
    upper = [np.inf, np.inf, np.inf, np.inf, np.inf]
    return p0, lower, upper


def _run_fit(
    model_func,
    t: np.ndarray,
    y: np.ndarray,
    p0: list,
    bounds: Tuple[list, list],
    param_names: list,
    model_name: str,
) -> SingleFitResult:
    """Execute a single curve_fit and package the results."""
    n_points = len(t)
    n_params = len(param_names)

    try:
        popt, pcov = curve_fit(
            model_func, t, y, p0=p0, bounds=bounds, maxfev=10000
        )
    except (RuntimeError, ValueError) as e:
        logger.warning(f"{model_name} fit failed: {e}")
        return SingleFitResult(
            success=False,
            model_name=model_name,
            error_message=f"Fit did not converge: {str(e)}",
            n_points=n_points,
            n_params=n_params,
        )
    except Exception as e:
        logger.error(f"{model_name} fit raised unexpected exception: {e}", exc_info=True)
        return SingleFitResult(
            success=False,
            model_name=model_name,
            error_message=f"Unexpected error: {str(e)}",
            n_points=n_points,
            n_params=n_params,
        )

    # Extract standard errors from covariance diagonal
    try:
        perr = np.sqrt(np.diag(pcov))
    except (ValueError, FloatingPointError):
        perr = np.full(n_params, np.nan)

    # Sanitize non-finite stderrs (covariance can be singular even on nominal success)
    perr = np.where(np.isfinite(perr), perr, np.nan)

    params = {name: float(val) for name, val in zip(param_names, popt)}
    param_stderr = {name: float(err) for name, err in zip(param_names, perr)}

    y_pred = model_func(t, *popt)
    r2, adj_r2, rmse, ss_res, aic, bic = _compute_fit_stats(y, y_pred, n_params)

    return SingleFitResult(
        success=True,
        model_name=model_name,
        error_message=None,
        params=params,
        param_stderr=param_stderr,
        r_squared=r2,
        adjusted_r_squared=adj_r2,
        rmse=rmse,
        ss_res=ss_res,
        aic=aic,
        bic=bic,
        n_points=n_points,
        n_params=n_params,
    )


# --- Public API ---

def _failure_result(
    error_message: str,
    range_start_ms: float,
    range_end_ms: float,
    time_fit_ms: Optional[np.ndarray] = None,
    current_fit: Optional[np.ndarray] = None,
    auto_detected: bool = False,
) -> KineticsResult:
    """Construct a failure KineticsResult with consistent field population."""
    return KineticsResult(
        success=False,
        error_message=error_message,
        direction="unknown",
        range_start_ms=range_start_ms,
        range_end_ms=range_end_ms,
        fit_region_start_ms=range_start_ms,
        fit_region_end_ms=range_end_ms,
        auto_detected=auto_detected,
        time_fit_ms=time_fit_ms if time_fit_ms is not None else np.array([]),
        current_fit=current_fit if current_fit is not None else np.array([]),
        mono=None,
        biexp=None,
        delta_aic=None,
        delta_bic=None,
    )


def fit_kinetics(
    time_ms: np.ndarray,
    current: np.ndarray,
    range_start_ms: float,
    range_end_ms: float,
    auto_detect_region: bool = False,
) -> KineticsResult:
    """
    Fit mono- and bi-exponential models to a current trace over [range_start_ms, range_end_ms].

    Args:
        time_ms: Full sweep time array in milliseconds.
        current: Current samples aligned with time_ms.
        range_start_ms: Start of user-specified fit window (ms).
        range_end_ms: End of user-specified fit window (ms).
        auto_detect_region: If True, narrow the fit to a sub-window anchored
            at the peak (decaying) or trough (rising) of the trace inside the
            user range. If False (default), use the full user range.

    Returns:
        KineticsResult with both fits, comparison metrics, and the actual
        fit region used. On irrecoverable input errors (empty window, too few
        points), .success is False and .error_message explains why.
    """
    if time_ms is None or current is None:
        return _failure_result(
            "Time or current data is None.",
            range_start_ms, range_end_ms,
            auto_detected=False,
        )

    if range_end_ms <= range_start_ms:
        return _failure_result(
            "Range end must be greater than range start.",
            range_start_ms, range_end_ms,
            auto_detected=False,
        )

    # Mask to user-specified window
    mask = (time_ms >= range_start_ms) & (time_ms <= range_end_ms)
    t_masked = time_ms[mask]
    y_masked = current[mask]

    # Drop NaN/inf samples
    finite = np.isfinite(t_masked) & np.isfinite(y_masked)
    t_masked = t_masked[finite]
    y_masked = y_masked[finite]

    n = len(t_masked)
    if n < 10:
        return _failure_result(
            f"Not enough valid samples in fit window ({n} found, need >= 10). "
            "Widen Range 1 or check the data.",
            range_start_ms, range_end_ms,
            time_fit_ms=t_masked,
            current_fit=y_masked,
            auto_detected=False,
        )

    # Auto-detect direction based on the full user range (more robust than the
    # potentially narrow sub-region)
    direction = _detect_direction(y_masked)

    # Optionally narrow to peak/trough-anchored sub-region
    auto_detected_effective = False
    if auto_detect_region:
        t_sub, y_sub, start_idx = _auto_detect_subregion(t_masked, y_masked, direction)
        if start_idx > 0:
            t_masked = t_sub
            y_masked = y_sub
            auto_detected_effective = True
        # If _auto_detect_subregion returned start_idx == 0, it already logged
        # the fallback; we proceed with the full range and auto_detected stays False.

    fit_region_start_ms = float(t_masked[0])
    fit_region_end_ms = float(t_masked[-1])

    # Shift time so t=0 is at the fit region start (in ms)
    t_fit = t_masked - fit_region_start_ms

    if direction == "rising":
        mono_func = _mono_rising
        biexp_func = _biexp_rising
    else:
        mono_func = _mono_decaying
        biexp_func = _biexp_decaying

    # Mono-exp fit
    p0_m, lo_m, hi_m = _initial_guesses_mono(t_fit, y_masked, direction)
    mono_result = _run_fit(
        mono_func, t_fit, y_masked, p0_m, (lo_m, hi_m),
        param_names=["A", "tau", "C"],
        model_name="Mono-exponential",
    )

    # Bi-exp fit
    p0_b, lo_b, hi_b = _initial_guesses_biexp(t_fit, y_masked, direction)
    biexp_result = _run_fit(
        biexp_func, t_fit, y_masked, p0_b, (lo_b, hi_b),
        param_names=["A1", "tau1", "A2", "tau2", "C"],
        model_name="Bi-exponential",
    )

    # Comparison metrics (only if both succeeded)
    if mono_result.success and biexp_result.success:
        delta_aic = biexp_result.aic - mono_result.aic
        delta_bic = biexp_result.bic - mono_result.bic
    else:
        delta_aic = None
        delta_bic = None

    logger.info(
        f"Kinetics fit complete: direction={direction}, n={len(t_fit)}, "
        f"auto_detected={auto_detected_effective}, "
        f"fit_region=[{fit_region_start_ms:.3f}, {fit_region_end_ms:.3f}] ms, "
        f"mono_success={mono_result.success}, biexp_success={biexp_result.success}"
    )

    return KineticsResult(
        success=True,
        error_message=None,
        direction=direction,
        range_start_ms=range_start_ms,
        range_end_ms=range_end_ms,
        fit_region_start_ms=fit_region_start_ms,
        fit_region_end_ms=fit_region_end_ms,
        auto_detected=auto_detected_effective,
        time_fit_ms=t_fit,
        current_fit=y_masked,
        mono=mono_result,
        biexp=biexp_result,
        delta_aic=delta_aic,
        delta_bic=delta_bic,
    )