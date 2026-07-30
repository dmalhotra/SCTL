#!/bin/bash

# Set up batch job settings
#SBATCH --job-name=sphere_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --constraint=icelake
#SBATCH --time=03:00:00

# Example slurm script for accuracy + timing results on a sphere using quad-element.

WORK_DIR=~/SCTL_quad_element
source ${WORK_DIR}/sctl_source

cd ${WORK_DIR}
make bin/test-quad-elem -j || exit 1

# Run test-quad-elem with a given thread count, pinned to a specific core range.
#   $1 = number of cores/threads
#   $2 = taskset core list (e.g. "0-3")
#   $3 = output file
run_sweep() {
    local ncores=$1
    local cpulist=$2
    local outfile=$3
    OMP_NUM_THREADS=${ncores} \
        taskset -c ${cpulist} \
        mpirun -n 1 --map-by :PE=${ncores} \
        ${WORK_DIR}/bin/test-quad-elem > ${outfile}
}

# Wave 1: 1-core (1), 4-core (2-5), 8-core (6-13), 16 core (14-29)
run_sweep 1  0-1  ${WORK_DIR}/results_spheresweep20_1core.txt  &
run_sweep 4  1-4  ${WORK_DIR}/results_spheresweep20_4core.txt  &
run_sweep 8  5-12  ${WORK_DIR}/results_spheresweep20_8core.txt  &
run_sweep 16 13-28 ${WORK_DIR}/results_spheresweep20_16core.txt &
run_sweep 32 32-64 ${WORK_DIR}/results_spheresweep20_32core.txt
