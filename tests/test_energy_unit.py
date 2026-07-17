"""--energy_unit: opt-in meV -> K conversion in the summation kernels.

The kernels are unit-agnostic by design (kB = 1: temperatures and
energies on the same scale); a hardcoded meV->K multiply would corrupt
every K-unit run (the pyrochlore fit pipeline feeds couplings in K).
The conversion is therefore an EXPLICIT flag, applied identically in
both kernels, and pinned here against the analytic two-level Schottky
form: Cv(T) = (d/T)^2 e^{d/T} / (1 + e^{d/T})^2 with d = Delta/kB in K.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from qed_nlce.run.NLC_sum import NLCExpansion  # noqa: E402
from qed_nlce.run.NLC_sum_triangular import NLCExpansionTriangular  # noqa: E402

KB_MEV_PER_K = 8.617333262e-2
DELTA_MEV = 0.2                      # two-level splitting, meV
DELTA_K = DELTA_MEV / KB_MEV_PER_K   # = 2.3209... K


def _schottky(T, d):
    x = d / T
    return x**2 * np.exp(x) / (1.0 + np.exp(x)) ** 2


def _mk(kernel_cls, tmp_path, **kw):
    d = tmp_path / "x"
    d.mkdir(exist_ok=True)
    if kernel_cls is NLCExpansion:
        return kernel_cls(str(d), str(d), 0.5, 20.0, 60, False, **kw)
    return kernel_cls(str(d), str(d), temp_min=0.5, temp_max=20.0,
                      num_temps=60, **kw)


@pytest.mark.parametrize("kernel_cls", [NLCExpansion, NLCExpansionTriangular])
def test_mev_eigenvalues_give_schottky_in_kelvin(kernel_cls, tmp_path):
    nlc = _mk(kernel_cls, tmp_path, energy_unit="meV")
    ev_mev = np.array([-DELTA_MEV / 2, +DELTA_MEV / 2])
    out = nlc.calculate_thermodynamic_quantities(ev_mev)
    ref = _schottky(nlc.temp_values, DELTA_K)
    np.testing.assert_allclose(out["specific_heat"], ref, atol=1e-12)


@pytest.mark.parametrize("kernel_cls", [NLCExpansion, NLCExpansionTriangular])
def test_default_is_unit_agnostic_unchanged(kernel_cls, tmp_path):
    """The default MUST remain the historical kB=1 behaviour: eigenvalues
    already on the temperature scale, no conversion."""
    nlc = _mk(kernel_cls, tmp_path)
    assert nlc.e_scale == 1.0
    ev_k = np.array([-DELTA_K / 2, +DELTA_K / 2])
    out = nlc.calculate_thermodynamic_quantities(ev_k)
    ref = _schottky(nlc.temp_values, DELTA_K)
    np.testing.assert_allclose(out["specific_heat"], ref, atol=1e-12)


@pytest.mark.parametrize("kernel_cls", [NLCExpansion, NLCExpansionTriangular])
def test_k_alias_matches_default(kernel_cls, tmp_path):
    a = _mk(kernel_cls, tmp_path, energy_unit="K")
    assert a.e_scale == 1.0


def test_rejects_unknown_unit(tmp_path):
    with pytest.raises(ValueError):
        _mk(NLCExpansion, tmp_path, energy_unit="eV")
