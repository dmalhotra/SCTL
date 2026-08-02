#!/bin/bash
# Strong-scaling + convergence run of bench-cubed-sphere on an Icelake node.
#
# Submit with:      sbatch scripts/bench-cubed-sphere-icelake.sbatch
# Watch with:       squeue -u $USER
# Efficiency after: seff <jobid>
#
# NOTE ON -march=native. The Makefile compiles with -march=native, so a binary built on
# Icelake contains AVX-512 and will die with SIGILL on the Zen 2 workstations, and vice
# versa. This script therefore builds into per-architecture directories (bin.icelake /
# obj.icelake) and never touches ./bin, so the two builds cannot clobber each other in
# the shared home filesystem.

#SBATCH --job-name=quadelem-icelake
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=10080
#SBATCH --output=doc/data/icelake-%j.out
#SBATCH --error=doc/data/icelake-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p doc/data

echo "host      : $(hostname)"
echo "cpu       : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "avx512f   : $(grep -c -m1 avx512f /proc/cpuinfo || true)"
echo "cores/task: ${SLURM_CPUS_PER_TASK:-unset}"
echo

# Build for THIS architecture, in its own tree.
make -j16 BINDIR=./bin.icelake OBJDIR=./obj.icelake bin.icelake/bench-cubed-sphere

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# No arguments: the full convergence sweep, then the strong-scaling descent from
# OMP_NUM_THREADS down to 1. The driver warns if the threads do not land on distinct cores.
./bin.icelake/bench-cubed-sphere | tee "doc/data/icelake-${SLURM_CPUS_PER_TASK:-64}.txt"
