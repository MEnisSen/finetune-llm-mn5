#!/bin/bash
#SBATCH --job-name=finetune_mdl
#SBATCH --account=ACCOUNT_NAME      #<--- account name
#SBATCH --qos=QOS_NAME              #<--- qos name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

export PROJECT_ROOT="project/root/path"

# Required by MN5: srun does not inherit cpus-per-task without this
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

module load miniforge
source activate venv/name

srun bash -c "
  torchrun \
    --nproc_per_node=4 \
    ${PROJECT_ROOT}/finetune_model.py
"