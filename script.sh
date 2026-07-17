#!/bin/bash

# Set up batch job settings
#SBATCH --job-name=job-name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00

# Example slurm script for accuracy + timing results on a sphere using quad-element.

WORK_DIR=~/SCTL_quad_element
source ${WORK_DIR}/sctl_source

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ${WORK_DIR}
make bin/test-quad-elem -j &&
mpirun -n 1 --map-by :PE=${OMP_NUM_THREADS} ${WORK_DIR}/bin/test-quad-elem > ${WORK_DIR}/results_1core.txt