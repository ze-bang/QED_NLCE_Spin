#!/usr/bin/env bash
# =============================================================================
# Production NLCE fit — NdMgAl11O19, triangle-based expansion (default in
# nlc_fit_triangular.py cluster generator: generate_triangle_nlce_clusters.py).
#
# Compares Hamiltonian stacks:
#   MODEL=xxz_j1j2     →  J1–J2 (+ optional --fit_Jz_ratio for XXZ anisotropy)
#   MODEL=anisotropic  →  bond-directional Jzz,Jpm,Jpmpm,Jzpm
#
# Data / strategy (see ../docs/FITTING_STRATEGY.txt):
#   * Strongly recommended: EXP_CONFIG=../configs/NdMgAl_production_0T_1T.json
#     (0 T tail + 1 T Schottky — much more discriminative than 0 T alone).
#   * Conservative single-field: NdMgAl_production_0T_only.json
#
# Wall-time / CPU (for queue planning)
# ------------------------------
# skopt gp_minimize uses n_calls = BO_MAX total objective evaluations
# (includes n_initial random points; see nlc_fit_triangular.py).
# Per-evaluation wall at ORDER=8 is dominated by FULL ED + wynn_multi NLCE;
# your stage-2 notes used ~27–35 min/eval on 96 cores × 16 parallel ED
# × OMP=6.  Rough wall ≈ BO_MAX * t_eval (parallel_fields runs 0T and 1T
# NLCE concurrently, so t_eval ≈ max of the two, not the sum).
#
#   BO_MAX=40  → ~18–28 h   (good first pass, shorter queue wait)
#   BO_MAX=64  → ~29–45 h   (request --time=48:00:00–60:00:00)
#   BO_MAX=200 → ~90–120 h (request --time=120:00:00 or split into 2 jobs)
#
# CPU-hours (Allied billing / footprint): ~96 cpus * wall_hours
# (e.g. 48 h × 96 ≈ 4600 CPU-h per job).
#
# Typical launch (override BO_MAX / account if needed):
#   cd /scratch/zhouzb79
#   mkdir -p NdMgAl_NLCE/logs
#   sbatch --account=def-ybkim --chdir=/scratch/zhouzb79 \
#     --output=NdMgAl_NLCE/logs/prod_%j.out --error=NdMgAl_NLCE/logs/prod_%j.err \
#     NdMgAl_NLCE/scripts/run_production_nlce_triangular_fit.sh
#
# Environment overrides:
#   MODEL=xxz_j1j2|anisotropic
#   ANISO_MODE=xxz_subspace|full     (only for MODEL=anisotropic)
#   FIT_JZ_RATIO=0|1                 (only for MODEL=xxz_j1j2 — adds Jz_ratio)
#   ORDER=8  BO_MAX=200  BO_INIT=40  OMP_THREADS=6  ED_CORES=16
#   STREAMING=0|1   ED_METHOD=AUTO|FULL|FTLM  RESUM=wynn_multi|euler
#   QED_NLCE_ROOT=...   SCRATCH=/scratch/zhouzb79
# =============================================================================
#SBATCH --account=def-ybkim
#SBATCH --chdir=/scratch/zhouzb79
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=60:00:00
#SBATCH --mem=384G
#SBATCH --job-name=ndmgal_nlce
#SBATCH --output=NdMgAl_NLCE/logs/prod_%j.out
#SBATCH --error=NdMgAl_NLCE/logs/prod_%j.err

set -euo pipefail

: "${SCRATCH:=/scratch/zhouzb79}"
: "${QED_NLCE_ROOT:=${HOME}/links/projects/def-ybkim/zhouzb79/QED_NLCE}"
: "${MODEL:=anisotropic}"                       # xxz_j1j2 | anisotropic
: "${ANISO_MODE:=xxz_subspace}"                 # xxz_subspace | full
: "${FIT_JZ_RATIO:=0}"                          # 0 | 1  (xxz_j1j2 only)
: "${ORDER:=8}"
# Default BO budget targets ~2× your O8 refine job (40 calls) for similar
# wall; raise BO_MAX for a full 200-call survey (use longer --time).
: "${BO_MAX:=72}"
: "${BO_INIT:=24}"
: "${OMP_THREADS:=6}"
: "${ED_CORES:=16}"
: "${STREAMING:=0}"
# FULL is safest on CPU-only `qed` wheels; use AUTO on GPU nodes after
# `pip install -e ./QED` with WITH_CUDA=ON.
: "${ED_METHOD:=FULL}"
: "${RESUM:=wynn_multi}"
: "${SYMM_THRESH:=9}"
: "${EXP_CONFIG:=${SCRATCH}/NdMgAl_NLCE/configs/NdMgAl_production_0T_1T.json}"
: "${ED_EXE:=/usr/bin/true}"

cd "${SCRATCH}" || { echo "ERROR: cannot cd to ${SCRATCH}"; exit 1; }

# --- Environment: same module stack as the legacy NdMgAl_phase1_order6.sh ---
if command -v module >/dev/null 2>&1; then
  module --force purge 2>/dev/null || true
  module load StdEnv/2023 2>/dev/null || true
  module load gcc/12.3 openmpi/4.1.5 2>/dev/null || true
  module load aocl-blas/5.1 aocl-lapack/5.1 2>/dev/null || true
  module load python scipy-stack mpi4py 2>/dev/null || true
  module load eigen arpack-ng 2>/dev/null || true
fi
export FLEXIBLAS=AOCL

# Activate the user venv that contains the editable `qed` and `qed_nlce`
# installs (see /home/zhouzb79/pystandard).
PYVENV="${PYVENV:-${HOME}/pystandard}"
if [[ -f "${PYVENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PYVENV}/bin/activate"
else
  echo "ERROR: venv not found at ${PYVENV}; install qed + qed_nlce there first."
  exit 1
fi

export OMP_NUM_THREADS="${OMP_THREADS}"

if [[ -f "${QED_NLCE_ROOT}/qed_nlce/analysis/nlc_fit_triangular.py" ]]; then
  FITTER="${QED_NLCE_ROOT}/qed_nlce/analysis/nlc_fit_triangular.py"
else
  FITTER="$(python - <<'PY'
import importlib.util
s = importlib.util.find_spec("qed_nlce.analysis.nlc_fit_triangular")
assert s and s.origin
print(s.origin)
PY
)"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="${MODEL}"
[[ "${MODEL}" == anisotropic ]] && TAG+="_${ANISO_MODE}"
[[ "${MODEL}" == xxz_j1j2 && "${FIT_JZ_RATIO}" == 1 ]] && TAG+="_Jzratio"
RUN_ROOT="${SCRATCH}/NdMgAl_NLCE/runs/prod_${TAG}_o${ORDER}_${STAMP}"
OUT_DIR="${RUN_ROOT}/output"
WORK_DIR="${RUN_ROOT}/work"
mkdir -p "${OUT_DIR}" "${WORK_DIR}"

echo "============================================================"
echo "NdMgAl11O19  NLCE production fit  (triangle expansion)"
echo "  HOST:          $(hostname)"
echo "  MODEL:         ${MODEL}"
echo "  ANISO_MODE:    ${ANISO_MODE}"
echo "  FIT_JZ_RATIO:  ${FIT_JZ_RATIO}"
echo "  ORDER:         ${ORDER}"
echo "  BO:            n_calls=${BO_MAX}  n_initial=${BO_INIT}"
echo "  EXP_CONFIG:    ${EXP_CONFIG}"
echo "  FITTER:        ${FITTER}"
echo "  OUT_DIR:       ${OUT_DIR}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}  ED_CORES=${ED_CORES}"
echo "  ED_METHOD=${ED_METHOD}  RESUM=${RESUM}  STREAMING=${STREAMING}"
echo "  FLEXIBLAS=${FLEXIBLAS:-<unset>}  PYTHON=$(command -v python)"
echo "============================================================"

python - <<'PY'
import qed, qed_nlce
print("qed", qed.__version__, "CUDA" if qed.has_cuda_build() else "CPU-only")
PY

COMMON=(
  --exp_config "${EXP_CONFIG}"
  --output_dir "${OUT_DIR}"
  --work_dir "${WORK_DIR}"
  --ed_executable "${ED_EXE}"
  --max_order "${ORDER}"
  --temp_min 0.05
  --temp_max 2.0
  --temp_bins 200
  --SI_units
  --field_dir 0 0 1
  --g_ab 1.54
  --g_c 3.75
  --method bayesian
  --max_iter "${BO_MAX}"
  --n_initial "${BO_INIT}"
  --acq_func EI
  --bo_log_transform
  --bo_inject_x0
  --parallel_ed
  --ed_num_cores "${ED_CORES}"
  --save_snapshots
  --save_landscape
  --resummation "${RESUM}"
  --symm_threshold "${SYMM_THRESH}"
  --ed_method "${ED_METHOD}"
)

# Multi-dataset: run fields in parallel when 0T+1T JSON is used
NDS="$(EXP_CONFIG="${EXP_CONFIG}" python -c "import json,os; print(len(json.load(open(os.environ['EXP_CONFIG']))['datasets']))")"
if [[ "${NDS}" -ge 2 ]]; then
  COMMON+=( --parallel_fields --max_parallel_fields "${NDS}" )
fi

if [[ "${STREAMING}" == 1 ]]; then
  COMMON+=( --streaming-symmetry )
fi

if [[ "${MODEL}" == xxz_j1j2 ]]; then
  CMD=( python "${FITTER}" "${COMMON[@]}" --model xxz_j1j2
    --initial_J1 0.36 --initial_J2 0.0
    --J1_min 0.05 --J1_max 1.05 --J2_min -0.55 --J2_max 0.75 )
  if [[ "${FIT_JZ_RATIO}" == 1 ]]; then
    CMD+=( --fit_Jz_ratio --Jz_ratio 0.82 --Jz_ratio_min 0.35 --Jz_ratio_max 1.05 )
  else
    CMD+=( --Jz_ratio 1.0 )
  fi
elif [[ "${MODEL}" == anisotropic ]]; then
  CMD=( python "${FITTER}" "${COMMON[@]}" --model anisotropic
    --initial_Jzz 0.30 --initial_Jpm 0.25
    --Jzz_min 0.05 --Jzz_max 0.85 --Jpm_min 0.02 --Jpm_max 0.85 )
  if [[ "${ANISO_MODE}" == xxz_subspace ]]; then
    CMD+=( --initial_Jpmpm 0.0 --initial_Jzpm 0.0
           --Jpmpm_min 0.0 --Jpmpm_max 0.0 --Jzpm_min 0.0 --Jzpm_max 0.0 )
  else
    CMD+=( --initial_Jpmpm 0.02 --initial_Jzpm 0.02
           --Jpmpm_min -0.25 --Jpmpm_max 0.25 --Jzpm_min -0.25 --Jzpm_max 0.25 )
  fi
else
  echo "ERROR: MODEL must be xxz_j1j2 or anisotropic (got ${MODEL})"
  exit 2
fi

echo "Running:" "${CMD[@]}"
"${CMD[@]}"

echo "============================================================"
echo "Finished at $(date)"
echo "  ${OUT_DIR}/fit_results.json"
echo "  ${OUT_DIR}/nlc_fit_triangular.log"
echo "============================================================"
