#!/bin/bash -l
#SBATCH --account=des
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=256
#SBATCH --time=30:00
#SBATCH --output=JOB/JOB_OUT_check_%j.txt
#SBATCH --error=JOB/JOB_ERR_check_%j.txt
#SBATCH --mail-user acampos@cmu.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END

set -x

export HDF5_USE_FILE_LOCKING='FALSE'

module load python
conda activate sompz

srun -n 256 python assign_SOM_wide_balrog_mpi.py
