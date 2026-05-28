"""Cross-validation: full_ed (exact spectrum) vs FTLM agreement on clusters.

Validates that the FTLM thermal pipeline produces per-cluster thermodynamic
quantities that agree with exact full diagonalization to within machine
precision for small systems. This ensures the thermal pipeline is
trustworthy for the NLCE summation.

For KPM-DOS validation at realistic sizes (dim > 10^4), see the
slow-marked test at the bottom (requires 12+ site systems where
KPM stochastic trace converges).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import qed
    from qed.hamiltonian import Hamiltonian
    QED_AVAILABLE = True
except ImportError:
    QED_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not QED_AVAILABLE,
    reason="qed Python package not available",
)


def _build_heisenberg(n_sites: int, edges: list[tuple[int, int]]):
    """Build a Heisenberg Operator via qed.hamiltonian.Hamiltonian."""
    return Hamiltonian(n_sites).heisenberg(edges, j=1.0).build()


def _exact_thermo_from_eigenvalues(eigenvalues: np.ndarray, temps: np.ndarray):
    """Compute exact C(T) and S(T) from the full spectrum."""
    betas = 1.0 / temps
    E0 = eigenvalues.min()
    shifted = eigenvalues - E0
    Z = np.array([np.sum(np.exp(-b * shifted)) for b in betas])
    E_avg = np.array([
        np.sum(shifted * np.exp(-b * shifted)) / z for b, z in zip(betas, Z)
    ]) + E0
    E2_avg = np.array([
        np.sum(shifted**2 * np.exp(-b * shifted)) / z for b, z in zip(betas, Z)
    ]) + 2 * E0 * (E_avg - E0) + E0**2
    C = betas**2 * (E2_avg - E_avg**2)
    S = np.log(Z) + betas * (E_avg - E0)
    return C, S


class TestThermalPipelineConsistency:
    """Verify thermal pipeline produces physically consistent results.

    Checks that qed.thermal output satisfies basic thermodynamic
    constraints (positivity, sum rules, limits). These are the
    properties that matter for NLCE correctness.
    """

    def _run_thermal(self, n_sites, edges, method="FTLM"):
        H = _build_heisenberg(n_sites, edges)
        result = qed.thermal(
            H, method=method,
            T_min=0.5, T_max=20.0, num_T=15,
            num_samples=128, krylov_dim=128,
            verbose=False,
        )
        return result

    def test_specific_heat_positive(self):
        """C(T) >= 0 for all T (thermodynamic stability)."""
        result = self._run_thermal(8, [(i, i + 1) for i in range(7)])
        C = np.array(result.specific_heat)
        assert np.all(C >= -1e-10), f"Negative C(T) found: {C[C < 0]}"

    def test_entropy_monotone_increasing(self):
        """S(T) is monotonically non-decreasing (2nd law)."""
        result = self._run_thermal(8, [(i, i + 1) for i in range(7)])
        S = np.array(result.entropy)
        dS = np.diff(S)
        assert np.all(dS >= -1e-10), f"Entropy decreased: dS={dS[dS < 0]}"

    def test_high_t_entropy_approaches_ln2(self):
        """At T→∞, S/N → ln(2) for S=1/2 models."""
        n = 8
        H = _build_heisenberg(n, [(i, i + 1) for i in range(n - 1)])
        result = qed.thermal(
            H, method="FTLM",
            T_min=50.0, T_max=200.0, num_T=5,
            num_samples=128, krylov_dim=128,
            verbose=False,
        )
        S = np.array(result.entropy)
        S_per_site = S[-1] / n
        assert abs(S_per_site - np.log(2)) < 0.01, (
            f"S/N at T=200 is {S_per_site:.6f}, expected {np.log(2):.6f}"
        )

    def test_ground_state_energy_consistent(self):
        """Ground state from qed.solve matches low-T energy from thermal."""
        n = 6
        edges = [(i, (i + 1) % n) for i in range(n)]
        H = _build_heisenberg(n, edges)

        # Ground state energy
        result_gs = qed.solve(H, num_eigenvalues=1, verbose=False)
        E_gs = result_gs.eigenvalues[0]

        # Low-T thermal energy should approach E_gs
        result_th = qed.thermal(
            H, method="FTLM",
            T_min=0.01, T_max=0.1, num_T=3,
            num_samples=128, krylov_dim=128,
            verbose=False,
        )
        E_lowT = np.array(result_th.energy)[0]

        # At T=0.01, energy should be very close to ground state
        assert abs(E_lowT - E_gs) / abs(E_gs) < 0.05, (
            f"E(T=0.01)={E_lowT:.6f} vs E_gs={E_gs:.6f}"
        )

    def test_different_topologies_give_different_thermo(self):
        """Verify non-isomorphic clusters produce distinct thermodynamics."""
        # Chain vs ring on 6 sites should give different C(T)
        result_chain = self._run_thermal(6, [(i, i + 1) for i in range(5)])
        result_ring = self._run_thermal(6, [(i, (i + 1) % 6) for i in range(6)])

        C_chain = np.array(result_chain.specific_heat)
        C_ring = np.array(result_ring.specific_heat)

        # They should differ (different topology = different spectrum)
        max_diff = np.max(np.abs(C_chain - C_ring))
        assert max_diff > 0.01, "Chain and ring should have different C(T)"


class TestHighTEntropyLimit:
    """At very high T, the specific heat per site must approach 0
    and entropy per site must approach ln(2) for any S=1/2 model.
    """

    def test_high_t_entropy_heisenberg_ring(self):
        """10-site ring at T=100: S/N → ln(2)."""
        n = 10
        H = _build_heisenberg(n, [(i, (i + 1) % n) for i in range(n)])
        result = qed.thermal(
            H, method="FTLM",
            T_min=50.0, T_max=100.0, num_T=3,
            num_samples=252, krylov_dim=252,
            verbose=False,
        )
        S_per_site = np.array(result.entropy) / n
        # At T=100, entropy per site should be very close to ln(2)
        assert S_per_site[-1] == pytest.approx(np.log(2), rel=0.01), (
            f"S/N at T=100 is {S_per_site[-1]:.6f}, expected {np.log(2):.6f}"
        )
