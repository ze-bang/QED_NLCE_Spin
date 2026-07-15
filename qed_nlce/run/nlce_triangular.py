#!/usr/bin/env python3
"""Legacy compatibility shim -- triangular lattice, full / ScaLAPACK ED.

Preserved for downstream analysis scripts (e.g.
``analysis/nlc_convergence_triangular*.py``, ``analysis/nlc_fit*.py``)
that invoke this file by path. For new work, use the unified CLI
directly::

    # triangle-based expansion (the historical default)
    python -m qed_nlce \\
        --geometry=triangular_triangle --pipeline=full_ed --max_order=4 ...

    # site-based expansion (legacy --site_based)
    python -m qed_nlce \\
        --geometry=triangular_site --pipeline=full_ed --max_order=4 ...

The translation layer below maps the legacy flag set onto the
unified CLI's flags.

NLCE on the triangular lattice runs ONLY through the triangle-based
expansion (``triangular_triangle``, order = number of triangles: order 2
is two triangles, and so on), which is normalized PER SITE. The legacy
``--site_based`` flag is retired -- the site-based expansion converges
poorly on a frustrated lattice and now exists only as a correctness
oracle (see ``geometries/triangular_site.py``).
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qed_nlce.cli import main as unified_main  # noqa: E402


def _translate_argv(argv: list[str]) -> list[str]:
    """Map legacy nlce_triangular.py argv onto the unified CLI argv.

    Legacy callers (e.g. ``analysis/nlc_fit_triangular.py``) pass
    ``--method=FULL`` / ``--method=FTLM`` / ``--method=AUTO`` /
    ``--method=KPM_DOS``. We auto-promote the chosen pipeline:

    * ``--method=AUTO``     -> ``--pipeline=auto``  (smart default)
    * ``--method=KPM_DOS*`` -> ``--pipeline=kpm_dos``
    * ``--method=FTLM*``    -> ``--pipeline=ftlm``
    * everything else       -> ``--pipeline=full_ed`` (forwards --method)
    """
    # Triangle-based expansion is the ONLY NLCE path on the triangular lattice
    # (per-site normalized). --site_based is retired; fail loudly rather than
    # silently running a different expansion.
    if "--site_based" in argv:
        raise SystemExit(
            "--site_based is retired: NLCE on the triangular lattice runs only "
            "through the triangle-based expansion (order = #triangles), which "
            "is per-site normalized. The site-based expansion is kept solely as "
            "a correctness oracle (geometries/triangular_site.py)."
        )
    geometry = "triangular_triangle"

    method_token = ""
    for tok in argv:
        if tok.startswith("--method="):
            method_token = tok.split("=", 1)[1].upper()
            break
        if tok == "--method":
            i = argv.index(tok)
            if i + 1 < len(argv):
                method_token = argv[i + 1].upper()
            break

    if method_token == "AUTO":
        pipeline = "auto"
    elif method_token.startswith("KPM_DOS") or method_token == "KPM":
        pipeline = "kpm_dos"
    elif method_token.startswith("FTLM") or method_token.startswith("LTLM"):
        pipeline = "ftlm"
    else:
        pipeline = "full_ed"

    out = [f"--geometry={geometry}", f"--pipeline={pipeline}"]
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok == "--site_based":
            continue
        # Strip --method for AUTO / KPM_DOS / FTLM pipelines (they own
        # the method choice). full_ed accepts --method directly so we
        # forward it.
        if pipeline != "full_ed" and (tok.startswith("--method=") or tok == "--method"):
            if tok == "--method":
                skip_next = True
            continue
        out.append(tok)
    return out


def main() -> int:
    argv = _translate_argv(sys.argv[1:])
    return unified_main(argv)


if __name__ == "__main__":
    start = time.time()
    rc = main()
    print(f"\nTotal execution time: {(time.time() - start) / 60:.2f} minutes")
    sys.exit(rc)
