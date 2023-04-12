#!/bin/bash -l
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=256
#SBATCH --constraint=haswell
#SBATCH --time=30:00
#SBATCH --license=SCRATCH
#SBATCH --output=JOB/JOB_OUT_check_%j.txt
#SBATCH --error=JOB/JOB_ERR_check_%j.txt

set -x

export HDF5_USE_FILE_LOCKING='FALSE'

conda activate sompz

srun -n 256 python assign_SOM_wide_noshear_y6_mpi.py