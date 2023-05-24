#!/bin/bash -l
#SBATCH --partition=regular
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=256
#SBATCH --constraint=haswell
#SBATCH --time=12:00:00
#SBATCH --license=SCRATCH
#SBATCH --output=JOB/JOB_OUT_check_%j.txt
#SBATCH --error=JOB/JOB_ERR_check_%j.txt
#SBATCH --mail-user acampos@cmu.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END

set -x

export HDF5_USE_FILE_LOCKING='FALSE'

module load python
conda activate sompz

srun -n 256 python assign_SOM_wide_noshear_y6_mpi.py