"""Shared test configuration: pin WHICH qed build the suite tests.

A scikit-build editable finder on ``sys.meta_path`` (installed by
``pip install -e`` of the sibling QED checkout) outranks ``sys.path``
and silently redirects ``import qed`` to a stale site-packages build --
resolution has been observed to flip-flop between runs, and a mixed
import (source ``__init__.py`` + site-packages submodules) breaks
outright. When the sibling QED source tree has a built ``_core``, strip
the finder and pin the import there; a wrong resolution is an immediate
red instead of a green suite on the wrong binary.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

_QED_PY = REPO_DIR.parent / "QED" / "python"
if any((_QED_PY / "qed").glob("_core*.so")):
    sys.meta_path = [
        f for f in sys.meta_path
        if "editable" not in type(f).__module__.lower()
    ]
    # Force FRONT position: the path may already be present at a LOSING
    # position (behind site-packages, e.g. via a stale .pth), where a
    # membership-guarded insert would silently keep the wrong winner.
    sys.path = [p for p in sys.path if p != str(_QED_PY)]
    sys.path.insert(0, str(_QED_PY))
    try:
        import qed
    except ImportError:
        pass  # tests importorskip qed themselves
    else:
        _got = Path(qed.__file__).resolve().parent
        assert _got == (_QED_PY / "qed").resolve(), (
            f"qed resolved to {_got}, expected the source-tree build at "
            f"{_QED_PY / 'qed'} -- another finder/path won; refusing to "
            "test the wrong build."
        )
