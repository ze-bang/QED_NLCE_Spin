"""Localise the little-group (point_group=auto) stall in qed.full_spectrum.

Empirically, on the 19-site order-6 pyrochlore cluster `cluster_9`,
`qed.full_spectrum(..., point_group="auto")` never returns (>12 h), while the
same operator with `point_group="off"` (abelian lane) solves in ~1543 s. The
non-abelian little-group lane is supposed to REFINE the abelian blocks, so it
should be strictly faster; the inversion is the defect.

This reproducer does four things, in order:
  1. reads the exact operator the pipeline used (no rebuild);
  2. resolves the symmetry the same way the pipeline does;
  3. CONTROL: times the C++ little-group PLANNER (plan_only=True) -- if this is
     fast, the star/abelian factorisation itself is fine and the stall is in
     the SOLVE, not the planning;
  4. calls the hanging solve, with a faulthandler watchdog dumping the Python
     stack every 90 s. A frozen Python frame sitting on the `qed.full_spectrum`
     line means the spin is in C++ (read it with gdb, driven by the sbatch);
     a moving pure-Python frame means the spin is in `point_group_routing`.

PR_SET_PTRACER_ANY is set so the sbatch's gdb can attach under yama
ptrace_scope=1.
"""

import ctypes
import faulthandler
import logging
import os
import sys
import time
from pathlib import Path

# Let any same-uid process (the sbatch's gdb) ptrace us, even under yama=1.
PR_SET_PTRACER = 0x59616D61
PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value
try:
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    print("prctl(PR_SET_PTRACER_ANY) ok", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"prctl failed ({exc}); gdb attach may need ptrace_scope=0", flush=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout)

sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/QED_NLCE")

HAM = ("/lustre09/project/6003507/zhouzb79/ce2hf2o7_nlce/o6off_setA/"
       "hamiltonians_order_6/cluster_9_order_6")
N = 19
PYSTACK = Path(__file__).resolve().parent / "diag_pystack.txt"

print(f"PID={os.getpid()}", flush=True)

from qed_nlce.ed.io import read_qed_operator            # noqa: E402
from qed_nlce.ed.engine import (                         # noqa: E402
    resolve_cluster_symmetry, _abelian_and_residues,
)
import qed                                               # noqa: E402
from qed import _core                                    # noqa: E402

qop = read_qed_operator(HAM, N)
print(f"read operator: num_sites={qop.num_sites}", flush=True)

t0 = time.monotonic()
cs = resolve_cluster_symmetry(qop, verbose=True)
print(f"resolved symmetry in {time.monotonic()-t0:.2f}s: "
      f"|A|={cs.abelian_size} residue={cs.num_star_perms} "
      f"sz_conserved={cs.sz_conserved} sz_parity={cs.sz_parity} "
      f"time_reversal={cs.time_reversal} spin_flip={cs.spin_flip}", flush=True)

# ---- CONTROL: does the C++ little-group PLANNER return quickly? ----------
try:
    A, residues = _abelian_and_residues(cs, N)
    print(f"factorisation: |A|={len(A)} n_residues={len(residues)}", flush=True)
    t0 = time.monotonic()
    plan = dict(_core.little_group_full_spectrum(
        qop, A, residues,
        n_up=(N // 2 if cs.sz_conserved else -1),
        sz_parity=(0 if (cs.sz_parity and not cs.sz_conserved) else -1),
        spin_flip=-1, time_reversal=-1, plan_only=True))
    dims = [int(st["dim_k0"]) for st in plan.get("stars", [])]
    print(f"PLAN_ONLY returned in {time.monotonic()-t0:.2f}s: "
          f"{len(dims)} stars, max dim_k0={max(dims) if dims else 0}",
          flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"plan_only control raised: {exc!r}", flush=True)

# ---- arm the watchdog and trigger the hang ------------------------------
stack_file = open(PYSTACK, "w")
faulthandler.dump_traceback_later(90, repeat=True, file=stack_file)

print("CALLING qed.full_spectrum(point_group='auto') -- expected to hang; "
      f"Python stack dumps go to {PYSTACK}", flush=True)
t0 = time.monotonic()
res = qed.full_spectrum(
    qop, symmetry=cs.gens, spin_flip="auto", time_reversal="auto",
    point_group="auto", device="cpu",
)
print(f"RETURNED in {time.monotonic()-t0:.2f}s with "
      f"{len(res.eigenvalues)} eigenvalues", flush=True)
