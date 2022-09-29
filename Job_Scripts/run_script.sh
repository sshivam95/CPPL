#!/bin/sh
# SBATCH -J "run_cadical_only_8"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -A hpc-prf-pbrac
#SBATCH -t 1-00:00:00
#SBATCH -p long
#SBATCH -o /scratch/hpc-prf-pbrac/sshivam/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-pbrac/sshivam/clusterout/%x-%j

cd $PFS_FOLDER/CPPL/
module reset
module load system/singularity/3.10.2
export IMG_FILE=$PFS_FOLDER/CPPL/singularity/pbcppl.sif
export SCRIPT_FILE=$PFS_FOLDER/CPPL/CPPL_test_run.py

export DIRECTORY=$1
export SOLVER=$2
export TIMEOUT=$3

module list
singularity exec -B $PFS_FOLDER/CPPL/ --nv $IMG_FILE pipenv run python3 $SCRIPT_FILE -d=$DIRECTORY -s=$SOLVER -to=$TIMEOUT

exit 0
~