"""Thermodynamics from a full eigenvalue spectrum.

All NLCE observables this package needs (energy, specific heat, entropy,
free energy) are functions of the *complete* set of energy eigenvalues.
The symmetry-blocked solver concatenates every block's spectrum into one
array that is identical (as a multiset) to the brute-force dense
spectrum, so these routines are oblivious to how the spectrum was
obtained.
"""

from __future__ import annotations

import numpy as np

__all__ = ["thermodynamics", "ThermoResult"]


class ThermoResult:
    """Container for temperature-resolved thermodynamic curves."""

    __slots__ = (
        "temperatures",
        "energy",
        "specific_heat",
        "entropy",
        "free_energy",
        "log_Z",
    )

    def __init__(self, T, E, C, S, F, logZ):
        self.temperatures = T
        self.energy = E
        self.specific_heat = C
        self.entropy = S
        self.free_energy = F
        self.log_Z = logZ


def thermodynamics(
    eigenvalues: np.ndarray,
    temperatures: np.ndarray,
) -> ThermoResult:
    """Compute per-temperature thermodynamics from the full spectrum.

    Uses a numerically stable log-sum-exp around the per-temperature
    ground state. ``eigenvalues`` must contain *every* eigenvalue
    (with its multiplicity); ``temperatures`` is an array of T values.
    """
    E0 = float(np.min(eigenvalues))
    evals = np.asarray(eigenvalues, dtype=np.float64)
    T = np.asarray(temperatures, dtype=np.float64)
    beta = 1.0 / T

    # x[t, n] = -beta_t * (E_n - E0)
    x = -np.outer(beta, evals - E0)
    # log-sum-exp over states for each temperature
    xmax = np.max(x, axis=1, keepdims=True)
    w = np.exp(x - xmax)
    Z_shift = np.sum(w, axis=1)              # = Z * exp(-xmax) ... handled below
    log_Z = (xmax[:, 0] + np.log(Z_shift)) - beta * E0

    # moments of E
    p = w / Z_shift[:, None]                 # Boltzmann probabilities
    mean_E = p @ evals
    mean_E2 = p @ (evals * evals)
    var_E = mean_E2 - mean_E * mean_E

    C = var_E * (beta * beta)
    F = -T * log_Z
    S = (mean_E - F) / T

    return ThermoResult(T, mean_E, C, S, F, log_Z)
