"""
Core engine for savings ODE models (first-order and second-order LC2).
Optimized version with improved performance, error handling, and documentation.
"""

import numpy as np
import math
from typing import Union, Tuple, Optional
from enum import Enum


class DampingRegime(Enum):
    """Enumeration for damping regimes in LC2 systems."""

    UNDERDAMPED = "underdamped"
    CRITICAL = "critical"
    OVERDAMPED = "overdamped"


class ModelType(Enum):
    """Enumeration for first-order model types."""

    GROWTH = "growth"
    DECAY = "decay"


# Constants
EPSILON = 1e-14
CRITICAL_DAMPING_TOLERANCE = 1e-12


def first_order_closed_form(
    model: str, t: Union[np.ndarray, float], S0: float, r: float, A: float
) -> np.ndarray:
    """
    Compute closed-form solution for first-order ODE models.

    Args:
        model: Model type ('growth' or 'decay')
        t: Time array or scalar
        S0: Initial savings
        r: Interest rate
        A: Contribution rate

    Returns:
        Solution array

    Raises:
        ValueError: If model type is invalid
        TypeError: If parameters have incorrect types
    """
    # Input validation
    if not isinstance(model, str):
        raise TypeError("Model must be a string")
    if model not in [ModelType.GROWTH.value, ModelType.DECAY.value]:
        raise ValueError(
            f"Invalid model: use '{ModelType.GROWTH.value}' or '{ModelType.DECAY.value}'"
        )

    # Convert to numpy array for vectorized operations
    t = np.asarray(t, dtype=np.float64)

    # Handle zero interest rate case
    if abs(r) < EPSILON:
        return S0 + A * t

    # Vectorized computation
    A_over_r = A / r
    if model == ModelType.GROWTH.value:
        return (S0 + A_over_r) * np.exp(r * t) - A_over_r
    else:  # decay
        return (S0 - A_over_r) * np.exp(-r * t) + A_over_r


def _lc2_homogeneous_constants(
    S0: float,
    V0: float,
    zeta: float,
    wn: float,
    Sstar: float,
    particular_prime_at_0: float = 0.0,
) -> Tuple[DampingRegime, tuple]:
    """
    Calculate homogeneous solution constants for LC2 system.

    Args:
        S0: Initial savings
        V0: Initial velocity
        zeta: Damping ratio
        wn: Natural frequency
        Sstar: Steady-state value
        particular_prime_at_0: Derivative of particular solution at t=0

    Returns:
        Tuple of (regime, parameters)
    """
    if wn <= 0:
        raise ValueError("Natural frequency must be positive")
    if zeta < 0:
        raise ValueError("Damping ratio must be non-negative")

    y0 = S0 - Sstar
    v0 = V0 - particular_prime_at_0

    if zeta < 1.0 - CRITICAL_DAMPING_TOLERANCE:
        # Underdamped
        zeta_sq = zeta * zeta
        wd = wn * math.sqrt(1.0 - zeta_sq)
        A = y0
        B = (v0 + zeta * wn * A) / wd
        return (DampingRegime.UNDERDAMPED, (A, B, wd))

    elif abs(zeta - 1.0) <= CRITICAL_DAMPING_TOLERANCE:
        # Critically damped
        C1 = y0
        C2 = v0 + wn * C1
        return (DampingRegime.CRITICAL, (C1, C2))

    else:
        # Overdamped
        sqrt_term = math.sqrt(zeta * zeta - 1.0)
        s1 = wn * (-zeta + sqrt_term)
        s2 = wn * (-zeta - sqrt_term)

        # Avoid division by zero
        denominator = s2 - s1
        if abs(denominator) < EPSILON:
            raise ValueError("Invalid damping parameters leading to division by zero")

        C1 = y0 * s2 / denominator
        C2 = y0 - C1
        return (DampingRegime.OVERDAMPED, (C1, C2, s1, s2))


def lc2_step_analytic(
    t: Union[np.ndarray, float],
    S0: float,
    V0: float,
    Sstar: float,
    wn: float,
    zeta: float,
) -> np.ndarray:
    """
    Analytical solution for LC2 system with step input.

    Args:
        t: Time array or scalar
        S0: Initial savings
        V0: Initial velocity
        Sstar: Steady-state target
        wn: Natural frequency
        zeta: Damping ratio

    Returns:
        Solution array
    """
    regime, params = _lc2_homogeneous_constants(S0, V0, zeta, wn, Sstar)
    t = np.asarray(t, dtype=np.float64)

    # Precompute common terms
    if regime == DampingRegime.UNDERDAMPED:
        A, B, wd = params
        exp_term = np.exp(-zeta * wn * t)
        y = exp_term * (A * np.cos(wd * t) + B * np.sin(wd * t))

    elif regime == DampingRegime.CRITICAL:
        C1, C2 = params
        exp_term = np.exp(-wn * t)
        y = (C1 + C2 * t) * exp_term

    else:  # OVERDAMPED
        C1, C2, s1, s2 = params
        y = C1 * np.exp(s1 * t) + C2 * np.exp(s2 * t)

    return Sstar + y


def lc2_sin_analytic(
    t: Union[np.ndarray, float],
    S0: float,
    V0: float,
    Sstar: float,
    wn: float,
    zeta: float,
    F: float,
    Omega: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Analytical solution for LC2 system with sinusoidal input.

    Args:
        t: Time array or scalar
        S0: Initial savings
        V0: Initial velocity
        Sstar: Steady-state target
        wn: Natural frequency
        zeta: Damping ratio
        F: Forcing amplitude
        Omega: Forcing frequency

    Returns:
        Tuple of (solution, amplitude, phase)
    """
    if Omega < 0:
        raise ValueError("Forcing frequency must be non-negative")

    t = np.asarray(t, dtype=np.float64)

    # Compute frequency response
    wn_sq = wn * wn
    Omega_sq = Omega * Omega
    two_zeta_wn_Omega = 2.0 * zeta * wn * Omega

    denom = math.sqrt((wn_sq - Omega_sq) ** 2 + two_zeta_wn_Omega**2)

    if denom < EPSILON:
        raise ValueError("Resonance condition: denominator too small")

    Ahat = F / denom
    phi = math.atan2(two_zeta_wn_Omega, wn_sq - Omega_sq)

    # Initial condition for particular solution derivative
    Sp_prime_0 = Ahat * Omega * math.cos(phi)

    regime, params = _lc2_homogeneous_constants(S0, V0, zeta, wn, Sstar, Sp_prime_0)

    # Particular solution
    particular = Ahat * np.sin(Omega * t - phi)

    # Homogeneous solution
    if regime == DampingRegime.UNDERDAMPED:
        A, B, wd = params
        exp_term = np.exp(-zeta * wn * t)
        homogeneous = exp_term * (A * np.cos(wd * t) + B * np.sin(wd * t))

    elif regime == DampingRegime.CRITICAL:
        C1, C2 = params
        exp_term = np.exp(-wn * t)
        homogeneous = (C1 + C2 * t) * exp_term

    else:  # OVERDAMPED
        C1, C2, s1, s2 = params
        homogeneous = C1 * np.exp(s1 * t) + C2 * np.exp(s2 * t)

    return Sstar + homogeneous + particular, Ahat, phi


def settling_time(
    t: np.ndarray, S: np.ndarray, S_star: Optional[float], tol: float = 0.05
) -> float:
    """
    Calculate settling time for a signal within tolerance band.

    Args:
        t: Time array
        S: Signal array
        S_star: Target steady-state value
        tol: Tolerance as fraction of S_star

    Returns:
        Settling time or NaN if not found
    """
    if S_star is None or math.isnan(S_star):
        return float("nan")

    if not (0 < tol < 1):
        raise ValueError("Tolerance must be between 0 and 1")

    if len(t) != len(S):
        raise ValueError("Time and signal arrays must have same length")

    # Compute tolerance band
    abs_S_star = abs(S_star)
    tol_band = tol * abs_S_star
    low = S_star - tol_band
    high = S_star + tol_band

    # Find where signal is within tolerance
    inside = (S >= low) & (S <= high)

    # Use numpy operations for better performance
    inside_indices = np.where(inside)[0]

    if len(inside_indices) == 0:
        return float("nan")

    # Check for continuous settling
    for i, idx in enumerate(inside_indices):
        if np.all(inside[idx:]):
            return float(t[idx])

    return float("nan")


# Additional utility functions
def compute_overshoot(S: np.ndarray, S_star: float) -> float:
    """
    Compute percentage overshoot relative to steady-state value.

    Args:
        S: Signal array
        S_star: Steady-state target value

    Returns:
        Percentage overshoot
    """
    if S_star == 0:
        return float("nan")

    max_val = np.max(S)
    return ((max_val - S_star) / abs(S_star)) * 100.0


def compute_rise_time(
    t: np.ndarray,
    S: np.ndarray,
    S_star: float,
    start_pct: float = 0.1,
    end_pct: float = 0.9,
) -> float:
    """
    Compute rise time between start_pct and end_pct of final value.

    Args:
        t: Time array
        S: Signal array
        S_star: Final steady-state value
        start_pct: Starting percentage (default 10%)
        end_pct: Ending percentage (default 90%)

    Returns:
        Rise time or NaN if not found
    """
    if S_star == 0:
        return float("nan")

    start_val = start_pct * S_star
    end_val = end_pct * S_star

    # Find crossing points
    start_idx = np.where(S >= start_val)[0]
    end_idx = np.where(S >= end_val)[0]

    if len(start_idx) == 0 or len(end_idx) == 0:
        return float("nan")

    return float(t[end_idx[0]] - t[start_idx[0]])
