#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --account=ACCOUNT_NAME      #<--- account name
#SBATCH --qos=QOS_NAME              #<--- qos name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepare_data_%j.out
#SBATCH --error=logs/prepare_data_%j.err

export PROJECT_ROOT="project/root/path"
export HF_DATASETS_OFFLINE=1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

module load miniforge
source activate venv/name

mkdir -p ${PROJECT_ROOT}/logs

srun python3 ${PROJECT_ROOT}/prepare_dataset_split.py
