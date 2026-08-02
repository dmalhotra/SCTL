#!/bin/bash
# Convergence sweep + OpenMP strong scaling for bench-cubed-sphere on one node.
#
# The architecture and width come from the submit line, so one script covers every node type:
#
#   sbatch -C icelake --cpus-per-task=64 scripts/bench-cubed-sphere.sbatch
#   sbatch -C genoa   --cpus-per-task=96 scripts/bench-cubed-sphere.sbatch
#   sbatch -C rome    --cpus-per-task=128 scripts/bench-cubed-sphere.sbatch
#
# Watch with `squeue -u $USER`; check the achieved efficiency afterwards with `seff <jobid>`.
#
# The driver's scaling section runs the full width and then powers of two down to 1, so a
# 96-core node gives 96, 64, 32, 16, 8, 4, 2, 1 -- comparable across machines with different
# core counts.
#
# NOTE ON -march=native. The Makefile compiles with -march=native, so an Icelake or Genoa
# binary carries AVX-512 and dies with SIGILL on the Zen 2 workstations, which share this home
# filesystem. Each architecture therefore builds into its own bin.<tag>/ and obj.<tag>/ and
# nothing ever writes ./bin.

#SBATCH --job-name=quadelem-scaling
#SBATCH --partition=ccm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=10080
#SBATCH --output=doc/data/slurm-%j.out
#SBATCH --error=doc/data/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p doc/data

NT="${SLURM_CPUS_PER_TASK:-$(nproc)}"
TAG="${SLURM_JOB_CONSTRAINTS:-}"
if [ -z "$TAG" ]; then
  TAG=$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //; s/[^A-Za-z0-9]/_/g' | cut -c1-20)
fi

echo "host       : $(hostname)"
echo "cpu        : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "sockets    : $(lscpu | awk -F: '/^Socket\(s\)/{print $2}' | tr -d ' ')"
echo "avx512f    : $(grep -c -m1 avx512f /proc/cpuinfo || true)"
echo "tag        : $TAG"
echo "threads    : $NT"
echo

make -j16 BINDIR="./bin.$TAG" OBJDIR="./obj.$TAG" "bin.$TAG/bench-cubed-sphere"

export OMP_NUM_THREADS="$NT"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# No arguments: convergence sweep, then the strong-scaling descent from $NT to 1.
# The driver warns if the threads do not land on distinct cores.
"./bin.$TAG/bench-cubed-sphere" | tee "doc/data/${TAG}-${NT}.txt"
