"""Bernu-Misguich entropy-method interpolation for NLCE output.

Extends thermodynamic curves below the NLCE convergence temperature by
constructing a smooth C_v(S) interpolant anchored at two exact limits:

  * High-T (reliable NLCE region): C_v(T) and S(T) from the NLCE sum.
  * Low-T (exact):  S → S_gs, C_v → 0, E → E0_per_site as T → 0.

The Bernu-Misguich trick (Bernu & Misguich, PRB 63, 134409, 2001) is to
change integration variable from T to S. The specific heat satisfies
dS = C_v(T)/T · d(ln T), so C_v(S) is a smooth function from C_v=0 at
S=S_gs to the NLCE values at high T. A PCHIP or polynomial interpolation
in S-space is better behaved than in T-space because the exponential
crowding near T=0 is removed.

Two integral sum rules constrain the interpolant:
  (1) Entropy:  ∫_{S_gs}^{S_∞} dS = S_∞ − S_gs
  (2) Energy:   ∫_0^∞ C_v(T) dT = E_∞ − E0_per_site

where E_∞ = 0 for normalised spin Hamiltonians (e(T→∞)=0) and
S_∞ = ln(2·spin + 1) per site.

References
----------
Bernu & Misguich, Phys. Rev. B 63, 134409 (2001).
Schmidt, Hauser, Lohmann, Richter, Phys. Rev. E 95, 042110 (2017).
Derzhko, Hutak, Krokhmalskii, Schnack, Richter, PRB 101, 174426 (2020)
  -- applied to spin-1/2 pyrochlore Heisenberg antiferromagnet.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def bernu_misguich(
    T_nlce: np.ndarray,
    Cv_nlce: np.ndarray,
    E0_per_site: float,
    S_gs: float = 0.0,
    spin: float = 0.5,
    T_conv_max: Optional[float] = None,
    n_out: int = 500,
) -> dict:
    """Interpolate thermodynamics below the NLCE convergence temperature.

    Parameters
    ----------
    T_nlce : ndarray, shape (N,)
        Temperature grid from the NLCE output (all temperatures, ascending).
    Cv_nlce : ndarray, shape (N,)
        Specific heat per site in units of k_B from NLCE.
    E0_per_site : float
        Ground-state energy per site in the same units as the Hamiltonian
        coupling J (e.g. J = 1). Required for the energy sum rule.
        Estimate from the highest-order NLCE cluster energy, or from a
        Lanczos calculation on a periodic finite cluster.
    S_gs : float, optional
        Ground-state entropy per site in units of k_B. Default 0 (non-
        degenerate ground state). For a pyrochlore spin liquid with
        residual entropy, set S_gs = k_B * ln(N_deg) / N_sites. If
        unknown, run with a range of values and compare to experiment.
    spin : float, optional
        Spin quantum number (default 0.5 for spin-1/2).
    T_conv_max : float, optional
        Use only NLCE data at T <= T_conv_max as the reliable high-T anchor.
        If None, all provided T_nlce data is used. For pyrochlore at order 8,
        typical value is 0.3–0.5 J.
    n_out : int, optional
        Number of temperature points in the returned interpolated curve.

    Returns
    -------
    dict with keys:
        'T'        : temperature grid (n_out points from near 0 to T_max_nlce)
        'Cv'       : interpolated C_v(T)
        'S'        : interpolated entropy S(T)
        'E'        : interpolated energy E(T) = E0 + ∫_0^T C_v dT'
        'F'        : free energy F(T) = E(T) − T·S(T)
        'S_nlce'   : entropy computed from NLCE C_v (input, before interp)
        'T_conv'   : highest T used as NLCE anchor
        'residual_entropy_check' : |∫C_v/T dT - (S_inf - S_gs)|/S_inf
        'energy_sum_rule_check'  : |∫C_v dT - (0 - E0)|/(|E0|+1e-12)
    """
    S_inf = np.log(2 * spin + 1)  # entropy per site at T=∞

    # --- 1. Select the reliable NLCE data window ---
    idx = np.argsort(T_nlce)
    T = T_nlce[idx]
    Cv = Cv_nlce[idx]

    if T_conv_max is not None:
        mask = T <= T_conv_max
        if mask.sum() < 3:
            raise ValueError(
                f"T_conv_max={T_conv_max} leaves fewer than 3 NLCE points. "
                "Lower T_conv_max or provide more data."
            )
        T = T[mask]
        Cv = Cv[mask]

    T_max = T[-1]

    # --- 2. Compute entropy S(T) by downward integration from T_max ---
    # S(T) = S(T_max) + ∫_{T_max}^{T} Cv(T')/T' d(ln T')
    # S(T_max) estimated by integrating Cv from very high T using the
    # high-T limit S(∞) = S_inf. We integrate the NLCE curve from T[-1]
    # up to T[-1] (no extrapolation needed if T_max >> J), but since we
    # only have data up to T_max, we approximate:
    #   S_residual_above_T_max = ∫_{T_max}^{∞} Cv/T dT
    # For spin-1/2 at T >> J, Cv ~ A/T² so ∫_{T_max}^∞ Cv/T dT ~ A/(2T_max²).
    # We use S_inf - ∫_0^{T_max} Cv(T')/T' dT' but integrate numerically.

    # Trapezoidal integration of Cv/T over ln(T) grid
    log_T = np.log(T)
    dlog_T = np.diff(log_T)            # positive (ascending T)
    Cv_mid = 0.5 * (Cv[:-1] + Cv[1:])
    delta_S_pieces = Cv_mid * dlog_T   # ∫C_v/T dT over each interval

    # cumulative integral from T[0] upward
    cum_integral = np.concatenate([[0.0], np.cumsum(delta_S_pieces)])
    # S(T) = S(T[0]) + cum_integral[i]
    # We get S(T[0]) from the constraint S(∞) = S_inf:
    # S(∞) = S(T[0]) + ∫_{T[0]}^{∞} Cv/T dT
    # The ∫_{T_max}^∞ part is small for T_max >> J; approximate as 0.
    # This introduces a small error for T_max ~ J; user should use T_max >> J.
    total_nlce_integral = cum_integral[-1]
    S_at_T0 = S_inf - total_nlce_integral  # approximate
    S_nlce = S_at_T0 + cum_integral         # entropy at each T point

    # --- 3. Build C_v(S) table for interpolation ---
    # Append the exact T=0 anchor: (S=S_gs, Cv=0)
    # The NLCE gives us (S_nlce, Cv) at the high-T end.
    # Combine in ascending S order.
    S_table = np.concatenate([[S_gs], S_nlce])
    Cv_table = np.concatenate([[0.0], Cv])

    # If S_gs >= S_nlce[0] (anomalous), flag and abort gracefully.
    if S_nlce[0] <= S_gs:
        raise ValueError(
            f"Computed S(T_min_nlce)={S_nlce[0]:.4f} <= S_gs={S_gs:.4f}. "
            "Either T_conv_max is too low (series not yet converged) or "
            "S_gs is overestimated. Increase T_conv_max or decrease S_gs."
        )

    # PCHIP preserves monotonicity and shape, avoiding Gibbs oscillations.
    interp_Cv_of_S = PchipInterpolator(S_table, Cv_table, extrapolate=False)

    # --- 4. Reconstruct T(S) by integrating d(ln T)/dS = 1/C_v(S) ---
    # Start from the highest NLCE anchor (S_nlce[-1], T[-1]) and integrate
    # downward in S toward S_gs.
    n_fine = 5000
    S_fine = np.linspace(S_nlce[-1], S_gs, n_fine)  # descending S
    Cv_fine = interp_Cv_of_S(S_fine)
    Cv_fine = np.clip(Cv_fine, 1e-14, None)  # avoid division by zero near T=0

    # d(ln T) = dS / C_v(S)  [negative, since S decreasing → T decreasing]
    dS = np.diff(S_fine)                                 # negative
    Cv_mid_fine = 0.5 * (Cv_fine[:-1] + Cv_fine[1:])
    d_logT = dS / Cv_mid_fine

    logT_fine = np.log(T[-1]) + np.concatenate([[0.0], np.cumsum(d_logT)])
    T_fine = np.exp(logT_fine)  # descending T array (T[-1] → ~0)

    # --- 5. Build output on a uniform temperature grid ---
    T_out = np.linspace(T_fine[-1] * 1.01, T_max, n_out)
    Cv_out = np.interp(T_out, T_fine[::-1], Cv_fine[::-1])
    S_out = np.interp(T_out, T_fine[::-1], S_fine[::-1])

    # Energy: E(T) = E0 + ∫_0^T C_v(T') dT'
    # Integrate Cv_out from T_out[0] (≈0) upward
    dT_out = np.diff(T_out)
    Cv_mid_out = 0.5 * (Cv_out[:-1] + Cv_out[1:])
    E_increments = np.concatenate([[0.0], np.cumsum(Cv_mid_out * dT_out)])
    E_out = E0_per_site + E_increments

    F_out = E_out - T_out * S_out

    # --- 6. Sum rule checks ---
    entropy_integral = np.trapezoid(Cv_out / T_out, np.log(T_out))
    entropy_check = abs(entropy_integral - (S_inf - S_gs)) / (S_inf + 1e-12)

    energy_integral = np.trapezoid(Cv_out, T_out)
    energy_check = abs(energy_integral - (0.0 - E0_per_site)) / (abs(E0_per_site) + 1e-12)

    return {
        "T": T_out,
        "Cv": Cv_out,
        "S": S_out,
        "E": E_out,
        "F": F_out,
        "S_nlce": S_nlce,
        "T_nlce_used": T,
        "Cv_nlce_used": Cv,
        "T_conv": float(T[-1]),
        "S_gs_used": S_gs,
        "E0_used": E0_per_site,
        "residual_entropy_check": float(entropy_check),
        "energy_sum_rule_check": float(energy_check),
    }


# ---------------------------------------------------------------------------
# HDF5 I/O helpers
# ---------------------------------------------------------------------------

def load_nlce_result(h5_path: str, dataset: str = "nlce") -> dict:
    """Load NLCE thermodynamic output from an HDF5 file.

    Tries several common path layouts written by the summation kernels.

    Returns dict with keys 'T', 'Cv', 'S', 'E', 'F'.
    """
    keys_to_try = [
        (f"/{dataset}/temperatures", f"/{dataset}/specific_heat",
         f"/{dataset}/entropy", f"/{dataset}/energy", f"/{dataset}/free_energy"),
        ("/nlce/temperatures", "/nlce/specific_heat",
         "/nlce/entropy", "/nlce/energy", "/nlce/free_energy"),
        ("/thermodynamics/temperatures", "/thermodynamics/specific_heat",
         "/thermodynamics/entropy", "/thermodynamics/energy", "/thermodynamics/free_energy"),
    ]
    with h5py.File(h5_path, "r") as f:
        for T_key, Cv_key, S_key, E_key, F_key in keys_to_try:
            try:
                return {
                    "T": f[T_key][:],
                    "Cv": f[Cv_key][:],
                    "S": f[S_key][:],
                    "E": f[E_key][:],
                    "F": f[F_key][:],
                }
            except KeyError:
                continue
    raise KeyError(
        f"Could not find thermodynamic datasets in {h5_path}. "
        f"Tried {[k[0] for k in keys_to_try]}. "
        f"Available top-level groups: {list(f.keys())}"
    )


def save_interpolated_result(h5_path: str, result: dict, group: str = "bernu_misguich") -> None:
    """Write the Bernu-Misguich result into an HDF5 file."""
    with h5py.File(h5_path, "a") as f:
        if group in f:
            del f[group]
        grp = f.create_group(group)
        for key in ("T", "Cv", "S", "E", "F"):
            grp.create_dataset(key, data=result[key].astype(np.float64))
        # Store metadata / diagnostics
        grp.attrs["E0_per_site"] = result["E0_used"]
        grp.attrs["S_gs"] = result["S_gs_used"]
        grp.attrs["T_conv"] = result["T_conv"]
        grp.attrs["entropy_sum_rule_error"] = result["residual_entropy_check"]
        grp.attrs["energy_sum_rule_error"] = result["energy_sum_rule_check"]
        grp.attrs["method"] = "Bernu-Misguich entropy interpolation"
        grp.attrs["reference"] = "Bernu & Misguich, PRB 63, 134409 (2001)"


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_comparison(result: dict, original: dict, output_path: Optional[str] = None) -> None:
    """Plot NLCE vs Bernu-Misguich interpolation for C_v and S."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(original["T"], original["Cv"], "o", ms=3, label="NLCE raw", color="C0", alpha=0.6)
    ax.plot(result["T"], result["Cv"], "-", lw=2, label="BM interpolation", color="C1")
    ax.axvline(result["T_conv"], ls="--", color="gray", lw=1, label=f"T_conv={result['T_conv']:.3f}")
    ax.set_xlabel("T / J")
    ax.set_ylabel("C_v / k_B")
    ax.set_title("Specific heat")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(original["T"], original["S"], "o", ms=3, label="NLCE raw", color="C0", alpha=0.6)
    ax.plot(result["T"], result["S"], "-", lw=2, label="BM interpolation", color="C1")
    ax.axhline(result["S_gs_used"], ls=":", color="gray", lw=1,
               label=f"S_gs={result['S_gs_used']:.4f}")
    ax.axvline(result["T_conv"], ls="--", color="gray", lw=1)
    ax.set_xlabel("T / J")
    ax.set_ylabel("S / k_B")
    ax.set_title("Entropy")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Bernu-Misguich  |  E₀={result['E0_used']:.4f}  "
        f"ΔS_err={result['residual_entropy_check']:.2e}  "
        f"ΔE_err={result['energy_sum_rule_check']:.2e}",
        fontsize=9,
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Bernu-Misguich entropy interpolation: extend NLCE thermodynamics "
            "below the convergence temperature using ground-state anchors. "
            "Reads thermodynamic data from an NLCE HDF5 output file and writes "
            "the interpolated curves to the same file under /bernu_misguich/."
        )
    )
    parser.add_argument("h5_path", help="NLCE HDF5 output file")
    parser.add_argument("--E0", type=float, required=True,
                        help="Ground-state energy per site in units of J. "
                             "Estimate from the highest-order NLCE cluster energy "
                             "or a Lanczos calculation on a finite periodic cluster. "
                             "For the S=1/2 pyrochlore Heisenberg AFM, E0 ≈ -0.39 J "
                             "(Schäfer et al. PRB 102, 054408, 2020).")
    parser.add_argument("--S_gs", type=float, default=0.0,
                        help="Ground-state entropy per site in k_B. "
                             "Default 0 (non-degenerate). For frustrated systems with "
                             "residual entropy, set to ln(degeneracy)/N_sites. "
                             "If unknown, scan a range and compare to experiments.")
    parser.add_argument("--spin", type=float, default=0.5,
                        help="Spin quantum number (default 0.5 for spin-1/2).")
    parser.add_argument("--T_conv", type=float, default=None,
                        help="Use only NLCE data at T <= T_conv as the reliable "
                             "anchor. Default: use all provided data. For pyrochlore "
                             "order 8, typical T_conv ≈ 0.3–0.5 J.")
    parser.add_argument("--dataset", type=str, default="nlce",
                        help="HDF5 group containing the NLCE output (default: nlce).")
    parser.add_argument("--n_out", type=int, default=500,
                        help="Number of output temperature points (default 500).")
    parser.add_argument("--plot", action="store_true",
                        help="Show a comparison plot.")
    parser.add_argument("--plot_output", type=str, default=None,
                        help="Save plot to this path instead of displaying.")
    args = parser.parse_args()

    if not os.path.isfile(args.h5_path):
        parser.error(f"File not found: {args.h5_path}")

    print(f"Loading NLCE data from {args.h5_path} ...")
    original = load_nlce_result(args.h5_path, dataset=args.dataset)
    print(f"  {len(original['T'])} temperature points, "
          f"T ∈ [{original['T'].min():.4f}, {original['T'].max():.4f}]")

    print(f"\nRunning Bernu-Misguich interpolation:")
    print(f"  E0/site = {args.E0:.6f} J")
    print(f"  S_gs/site = {args.S_gs:.6f} k_B")
    print(f"  spin = {args.spin}")
    print(f"  T_conv = {args.T_conv}")

    result = bernu_misguich(
        original["T"],
        original["Cv"],
        E0_per_site=args.E0,
        S_gs=args.S_gs,
        spin=args.spin,
        T_conv_max=args.T_conv,
        n_out=args.n_out,
    )

    print(f"\nSum rule checks:")
    print(f"  Entropy: |∫Cv/T dT - (S_∞ - S_gs)| / S_∞ = "
          f"{result['residual_entropy_check']:.3e}")
    print(f"  Energy:  |∫Cv dT - (0 - E0)| / |E0| = "
          f"{result['energy_sum_rule_check']:.3e}")

    if result["residual_entropy_check"] > 0.05:
        print("  WARNING: Entropy sum rule error > 5%. "
              "Either T_conv is too low (NLCE not yet converged at T_conv) "
              "or the high-T limit of the provided data is insufficient. "
              "Try increasing T_max or decreasing T_conv.")

    print(f"\nSaving interpolated result to {args.h5_path}:/bernu_misguich/ ...")
    save_interpolated_result(args.h5_path, result)
    print("Done.")

    if args.plot or args.plot_output:
        plot_comparison(result, original, output_path=args.plot_output)


if __name__ == "__main__":
    main()
