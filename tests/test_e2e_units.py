"""End-to-end units contract: --SI_units output is J/(mol K) PER SITE.

Runs the REAL CLI (both geometries, both unit lanes) at order 2 and pins
the infinite-temperature entropy: S -> ln 2 per site (dimensionless) and
R ln 2 = 5.7628 J/(mol K) per site (SI). This is the invariant that
catches any per-site normalization error anywhere in the chain
(eigenvalues -> per-cluster thermo -> Moebius weights -> resummation ->
SI scaling): a wrong site normalization shifts it by an O(1) factor,
while finite-T truncation at T = 20 J costs well under 1%.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]

LN2 = float(np.log(2.0))
R = 8.314462618  # J/(mol K)


def _run(base_dir, geometry, extra):
    cmd = [
        sys.executable, "-m", "qed_nlce",
        f"--geometry={geometry}", "--pipeline=full_ed", "--max_order=2",
        f"--base_dir={base_dir}", "--thermo", "--no_cache",
        "--temp_min=0.05", "--temp_max=20", "--temp_bins=60",
    ] + extra
    r = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, (
        f"CLI failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    hits = list(Path(base_dir).glob("nlc_results_order_*/nlc_entropy.txt"))
    assert hits, "no nlc_entropy.txt produced"
    d = np.loadtxt(hits[0])
    T, S = d[:, 0], d[:, 1]
    return float(S[np.argmax(T)])


@pytest.mark.parametrize("geometry,model_args", [
    ("triangular_triangle",
     ["--model=xxz_j1j2", "--J1=1.0", "--Jz_ratio=1.0"]),
    ("pyrochlore", ["--Jzz=1.0", "--Jxx=0.1", "--Jyy=0.1"]),
])
def test_high_T_entropy_per_site(tmp_path, geometry, model_args):
    s_dimless = _run(tmp_path / "dimless", geometry, model_args)
    s_si = _run(tmp_path / "si", geometry, model_args + ["--SI_units"])
    assert abs(s_dimless / LN2 - 1) < 0.01, (
        f"{geometry}: S_inf/site = {s_dimless} != ln2 -- per-site "
        "normalization broken")
    assert abs(s_si / (R * LN2) - 1) < 0.01, (
        f"{geometry}: SI S_inf/site = {s_si} != R ln2 -- SI scaling or "
        "per-site normalization broken")
    # SI must be EXACTLY R x dimensionless (same run, same grid).
    assert abs(s_si / (R * s_dimless) - 1) < 1e-6
