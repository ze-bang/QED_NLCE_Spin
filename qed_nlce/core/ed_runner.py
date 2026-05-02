"""Canonical bridge from NLCE Python code to the C++ ``./ED`` binary.

Every NLCE pipeline goes through ``EDOptions`` → :func:`build_ed_command`
→ :func:`run_ed_subprocess`. Routing all `./ED` invocations through
this single module guarantees:

* CLI flag changes only need to be audited here.
* Method auto-promotion (``FULL`` → ``SCALAPACK_MIXED`` for large
  clusters) is centralized.
* The well-documented "ED crashed during cleanup but the HDF5 file
  is intact" failure mode is reconciled in one place.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

from .io import HAS_H5PY

if HAS_H5PY:
    import h5py  # noqa: F401  (only imported for the guarded re-check below)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _candidate_ed_paths() -> list[str]:
    """Ordered list of plausible locations for the ``ED`` binary.

    qed_nlce ships standalone -- it does not bundle ``ED`` itself --
    so we must locate the binary at runtime. Search order:

    1. ``$QED_ED_BINARY`` env var (explicit override).
    2. ``$ED_BINARY`` env var (legacy override).
    3. ``shutil.which("ED")`` (binary on ``$PATH``, typical for users
       who ran ``cmake --install`` or rely on a wheel that drops the
       binary in ``$VIRTUAL_ENV/bin``).
    4. ``./build/ED`` relative to the current working directory
       (typical when the user runs ``qed-nlce`` from a sibling QED
       checkout).
    5. ``../QED/build/ED``, ``../../QED/build/ED`` -- relative to a
       QED checkout sitting next to QED_NLCE.
    6. ``$VIRTUAL_ENV/bin/ED``.
    """
    out: list[str] = []
    for env_var in ("QED_ED_BINARY", "ED_BINARY"):
        v = os.environ.get(env_var)
        if v:
            out.append(v)
    on_path = shutil.which("ED")
    if on_path:
        out.append(on_path)
    cwd = os.getcwd()
    out.append(os.path.join(cwd, "build", "ED"))
    out.append(os.path.join(cwd, "..", "QED", "build", "ED"))
    out.append(os.path.join(cwd, "..", "..", "QED", "build", "ED"))
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        out.append(os.path.join(venv, "bin", "ED"))
    # Try the qed Python package's install root as a last resort.
    try:
        import qed as _qed  # type: ignore
        pkg_dir = os.path.dirname(os.path.abspath(_qed.__file__))
        out.append(os.path.join(pkg_dir, "..", "..", "..", "bin", "ED"))
        out.append(os.path.join(pkg_dir, "..", "bin", "ED"))
    except Exception:
        pass
    return out


def discover_ed_binary() -> Optional[str]:
    """Return the first existing/executable path from :func:`_candidate_ed_paths`."""
    for p in _candidate_ed_paths():
        p_abs = os.path.abspath(p)
        if os.path.isfile(p_abs) and os.access(p_abs, os.X_OK):
            return p_abs
    return None


# Best-effort default at import time (pure string for backwards-compat).
# May not exist on disk -- callers must check.
DEFAULT_ED_PATH = discover_ed_binary() or os.path.join(os.getcwd(), "build", "ED")


__all__ = [
    "DEFAULT_ED_PATH",
    "EDOptions",
    "build_ed_command",
    "discover_ed_binary",
    "run_ed_subprocess",
]


@dataclass
class EDOptions:
    """Knobs the NLCE pipelines forward into one ``./ED`` invocation.

    Each attribute corresponds to a specific ``./ED`` CLI flag (or to a
    decision rule made by :func:`build_ed_command`). Pipeline classes
    construct an ``EDOptions`` per cluster from the user's CLI args.

    Attributes:
        method: ED method name (``FULL``, ``FULL_GPU``, ``SCALAPACK_MIXED``,
            ``mTPQ``, ``cTPQ``, ``LANCZOS``, ``FTLM``, ...). The ``FULL``
            default may be auto-promoted to ``SCALAPACK_MIXED`` when the
            cluster is large; see :func:`build_ed_command`.
        eigenvalues: Value for ``--eigenvalues`` (``FULL``, ``LOWEST``,
            integer count, or ``None`` to omit the flag).
        spin_length: Local spin length (``--spin_length``).
        thermo: Whether to enable the ``--thermo`` block.
        temp_min/temp_max/temp_bins: Temperature grid for ``--thermo``.
        measure_spin: Forward ``--measure_spin``.
        symmetrized: Use ``--symmetrized`` (a stronger guarantee than ``--symm``).
        use_symm: When ``symmetrized`` is False, whether to use ``--symm``.
        streaming_symmetry: Use ``--streaming-symmetry`` (orbit basis cache).
        basis_cache_dir: Optional directory for the streaming-symmetry basis cache.
        samples: ``--samples`` (FTLM / TPQ).
        krylov_dim: ``--krylov_dim`` (FTLM / TPQ).
        extra_flags: Free-form additional argv entries appended at the end.
    """

    method: str = "FULL"
    eigenvalues: Optional[str] = "FULL"
    spin_length: float = 0.5
    thermo: bool = False
    temp_min: float = 0.001
    temp_max: float = 20.0
    temp_bins: int = 100
    measure_spin: bool = False
    symmetrized: bool = False
    use_symm: bool = True
    streaming_symmetry: bool = False
    basis_cache_dir: Optional[str] = None
    samples: Optional[int] = None
    krylov_dim: Optional[int] = None
    extra_flags: list[str] = field(default_factory=list)


def build_ed_command(
    ed_executable: str,
    ham_subdir: str,
    output_dir: str,
    num_sites: int,
    options: EDOptions,
    *,
    scalapack_threshold: int = 16,
    use_scalapack: bool = True,
) -> list[str]:
    """Translate an :class:`EDOptions` into a fully-formed ``./ED`` argv vector.

    Method selection rules (matching the drivers' historical behaviour):

      * ``options.method == 'FULL_GPU'`` -> respected verbatim.
      * FTLM-flavoured methods -> respected verbatim.
      * ``options.method == 'FULL'`` (the default) AND the cluster is
        large (``num_sites >= scalapack_threshold``) AND ScaLAPACK is
        enabled -> rewrite to ``SCALAPACK_MIXED``.
      * Otherwise: use ``options.method`` as-is.
    """
    method = options.method.upper()
    hilbert_dim = 2 ** num_sites

    if method == "FULL" and use_scalapack and num_sites >= scalapack_threshold:
        method = "SCALAPACK_MIXED"

    cmd: list[str] = [
        ed_executable,
        ham_subdir,
        f"--method={method}",
        f"--output={output_dir}/output",
        f"--num_sites={num_sites}",
        f"--spin_length={options.spin_length}",
    ]

    if options.eigenvalues:
        cmd.append(f"--eigenvalues={options.eigenvalues}")

    if options.symmetrized:
        cmd.append("--symmetrized")
    elif options.use_symm:
        cmd.append("--symm")

    if options.streaming_symmetry:
        cmd.append("--streaming-symmetry")
        if options.basis_cache_dir and os.path.isdir(options.basis_cache_dir):
            cmd.append(f"--basis-cache-dir={options.basis_cache_dir}")

    if options.measure_spin:
        cmd.append("--measure_spin")

    if options.samples is not None:
        cmd.append(f"--samples={options.samples}")
    if options.krylov_dim is not None:
        cmd.append(f"--krylov_dim={options.krylov_dim}")

    if options.thermo:
        cmd.extend([
            "--thermo",
            f"--temp_min={options.temp_min}",
            f"--temp_max={options.temp_max}",
            f"--temp_bins={options.temp_bins}",
        ])

    cmd.extend(options.extra_flags)

    logging.debug(
        "build_ed_command: cluster=%s sites=%d dim=2^%d=%d method=%s argv=%s",
        ham_subdir, num_sites, num_sites, hilbert_dim, method, cmd,
    )
    return cmd


def _output_dir_has_results(output_dir: str) -> bool:
    """Best-effort check that an HDF5 / legacy output file is salvageable.

    The ED binary occasionally crashes during MPI / GPU tear-down *after*
    successfully writing its HDF5 results file. This helper lets the
    drivers treat such crashes as success without losing the genuine
    failure mode (segfault that produces no output at all).
    """
    if not os.path.isdir(output_dir):
        return False
    h5_candidates = [
        os.path.join(output_dir, "ed_results.h5"),
        os.path.join(output_dir, "thermo", "ed_results.h5"),
    ]
    for path in h5_candidates:
        if os.path.exists(path):
            if HAS_H5PY:
                try:
                    import h5py as _h5
                    with _h5.File(path, "r") as f:
                        if "eigenvalues" in f and f["eigenvalues"].shape[0] == 0:
                            return False
                    return True
                except Exception:  # pragma: no cover - corrupt file
                    return False
            return True
    txt_candidates = [
        os.path.join(output_dir, "thermo", "thermo_data.txt"),
        os.path.join(output_dir, "thermo", "ftlm_thermo.txt"),
    ]
    return any(os.path.exists(p) for p in txt_candidates)


def run_ed_subprocess(
    cmd: list[str],
    *,
    output_root: str,
    timeout: Optional[int] = None,
    extra_env: Optional[dict[str, str]] = None,
    log_tag: str = "ED",
) -> bool:
    """Run a single ``./ED`` invocation and reconcile exit-code vs output.

    Returns True iff the run produced usable output (even when the
    binary itself exited non-zero), False otherwise. ``output_root``
    is the directory passed via ``--output=<output_root>/output`` so we
    can inspect it on failure.

    ``timeout`` defaults to env var ``NLCE_ED_TIMEOUT`` (1 hr) when None.
    """
    env = os.environ.copy()
    env["ED_PYTHON"] = sys.executable
    if extra_env:
        env.update(extra_env)

    if timeout is None:
        try:
            timeout = int(os.environ.get("NLCE_ED_TIMEOUT", 3600))
        except (TypeError, ValueError):
            timeout = 3600

    output_dir = os.path.join(output_root, "output")

    try:
        subprocess.run(cmd, check=True, capture_output=True, env=env, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        logging.error("[%s] subprocess timed out after %ds: %s", log_tag, timeout, cmd[0])
        return False
    except subprocess.CalledProcessError as e:
        if _output_dir_has_results(output_dir):
            logging.warning(
                "[%s] subprocess exited with code %d but usable output exists -- treating as success",
                log_tag, e.returncode,
            )
            return True
        if e.returncode == -11:  # SIGSEGV
            logging.error("[%s] subprocess crashed with SIGSEGV (no output produced)", log_tag)
        else:
            logging.error("[%s] subprocess failed: %s", log_tag, e)
        if e.stdout:
            stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else e.stdout
            logging.error("[%s] stdout: %s", log_tag, stdout)
        if e.stderr:
            stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else e.stderr
            logging.error("[%s] stderr: %s", log_tag, stderr)
        return False
