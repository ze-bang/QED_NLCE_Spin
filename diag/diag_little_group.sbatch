#!/bin/bash
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=249G --time=0:40:00
#SBATCH --exclusive
#SBATCH --job-name=lg_diag
#SBATCH --output=/lustre09/project/6003507/zhouzb79/QED_NLCE/diag/lg_diag_%j.log
set -uo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.11.5 aocl-blas/5.1 aocl-lapack/5.1 hdf5/1.14.6
export FLEXIBLAS=AOCL OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK QED_DEVICE=cpu QED_CF_DEVICE=cpu
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/scipy-stack/2024b/lib/python3.11/site-packages:/lustre09/project/6003507/zhouzb79/QED/python:/lustre09/project/6003507/zhouzb79/QED_NLCE
cd /lustre09/project/6003507/zhouzb79/QED_NLCE/diag
git -C /lustre09/project/6003507/zhouzb79/QED_NLCE rev-parse --short HEAD
echo "ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo '?')  gdb=$(command -v gdb)"

python3 -u diag_little_group.py > diag_stdout_${SLURM_JOB_ID}.txt 2>&1 &
PID=$!
echo "python PID=$PID"

# Wait until the process is past read+resolve+plan_only and into the hanging
# solve.  Poll stdout for the trigger line rather than guessing a fixed sleep.
for _ in $(seq 1 120); do
  grep -q "expected to hang" diag_stdout_${SLURM_JOB_ID}.txt 2>/dev/null && break
  kill -0 $PID 2>/dev/null || { echo "python exited before the hang trigger"; break; }
  sleep 5
done
echo "=== entered solve at $(date); letting it spin 60s before first dump ==="
sleep 60

for i in 1 2 3 4 5; do
  kill -0 $PID 2>/dev/null || { echo "python exited; stopping dumps"; break; }
  echo "############## NATIVE DUMP $i  $(date) ##############"
  echo "--- per-thread CPU / kernel wait-channel ---"
  ps -L -p $PID -o pid,tid,pcpu,stat,wchan:28,comm 2>/dev/null | head -80
  echo "--- gdb: all-thread native backtrace ---"
  gdb -p $PID -batch -nx \
      -ex 'set pagination off' -ex 'set width 0' \
      -ex 'info threads' \
      -ex 'thread apply all bt' 2>&1 | head -400
  echo "############## END DUMP $i ##############"
  sleep 90
done

echo "=== PYTHON-LEVEL STACK DUMPS (faulthandler) ==="
cat diag_pystack.txt 2>/dev/null | head -200

kill -9 $PID 2>/dev/null
echo "done $(date)"
