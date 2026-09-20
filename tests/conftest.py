"""Shared test configuration: pin WHICH qed build the suite tests.

A scikit-build editable finder on ``sys.meta_path`` (installed by ``pip install -e``
of a sibling qed checkout) outranks ``sys.path`` and silently redirects ``import qed``
to a stale site-packages build; resolution has been observed to flip-flop between runs,
and a mixed import (source ``__init__.py`` + site-packages submodules) breaks outright.
So the suite pins one build and refuses to run against anything else.

WHICH build, in order:

1. the caller's, when ``QED_CORE_DIR`` (or ``QED_PYTHONPATH``) is set -- the modern
   contract, where ``PYTHONPATH`` carries the source package and ``QED_CORE_DIR`` the
   built extension. A gate testing a tree must keep testing THAT tree;
2. otherwise a sibling SOURCE tree carrying a built ``_core``, current name first.

``QED`` is the RETIRED name of the library repo (now ``QED_Spin``). This file used to
pin that sibling unconditionally, so with a stale build still on disk the downstream
gate exercised the old library while reporting the new one -- green for code nobody had
touched. That is what the ordering above exists to prevent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _carries_core(base: Path) -> bool:
    return bool(list((base / "qed").glob("_core*.so")))


_CALLER_PINNED = bool(os.environ.get("QED_CORE_DIR") or os.environ.get("QED_PYTHONPATH"))
_QED_PY = None
if not _CALLER_PINNED:
    for _name in ("QED_Spin", "QED"):
        _cand = REPO_DIR.parent / _name / "python"
        if _carries_core(_cand):
            _QED_PY = _cand
            break

# The editable finder outranks sys.path either way, so strip it whenever one is present.
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in type(f).__module__.lower()]

if _QED_PY is not None:
    # Force FRONT position: the path may already be present at a LOSING position
    # (behind site-packages, e.g. via a stale .pth), where a membership-guarded
    # insert would silently keep the wrong winner.
    sys.path = [p for p in sys.path if p != str(_QED_PY)]
    sys.path.insert(0, str(_QED_PY))

try:
    import qed
except ImportError:
    pass  # tests importorskip qed themselves
else:
    _got = Path(qed.__file__).resolve().parent
    if _QED_PY is not None:
        assert _got == (_QED_PY / "qed").resolve(), (
            f"qed resolved to {_got}, expected the source-tree build at "
            f"{_QED_PY / 'qed'} -- another finder/path won; refusing to test the "
            "wrong build."
        )
    else:
        _want = os.environ.get("QED_PYTHONPATH") or ""
        assert not _want or _got == (Path(_want) / "qed").resolve(), (
            f"qed resolved to {_got}, but QED_PYTHONPATH pinned {_want}."
        )
