#!/bin/bash
#SBATCH --job-name=finetune_lora_sft
#SBATCH --account=ACCOUNT_NAME      #<--- account name
#SBATCH --qos=QOS_NAME              #<--- qos name
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=logs/finetune_lora_sft_%j.out
#SBATCH --error=logs/finetune_lora_sft_%j.err

export PROJECT_ROOT="project/root/path"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Required by MN5: srun does not inherit cpus-per-task without this
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# NCCL settings for MN5 H100 InfiniBand fabric
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=22
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

module load miniforge
source activate venv/name

mkdir -p ${PROJECT_ROOT}/logs ${PROJECT_ROOT}/results_lora_sft

srun bash -c "
  torchrun \
    --nproc_per_node=4 \
    ${PROJECT_ROOT}/finetune_model_lora+sft.py
"