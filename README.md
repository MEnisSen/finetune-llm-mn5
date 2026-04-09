# LLM Fine-Tuning on HPC (MareNostrum 5)

Fine-tuning `Qwen3.5-4B` on the `OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B` dataset across 4 GPUs on MareNostrum 5.

---

## Step 1: Environment Setup

Load modules and create a virtual environment on the login node:

```bash
module load miniforge
conda create -n finetune_env python=3.10 -y
conda activate finetune_env
pip install -r requirements.txt
```

---

## Step 2: Download Model and Dataset

Compute nodes have no internet access. Download everything on the login node first:

```bash
# Download model
huggingface-cli download Qwen/Qwen3.5-4B \
  --local-dir /gpfs/projects/ehpc463/llm_finetuning/models/Qwen3.5-4B

# Download and split dataset
python3 prepare_dataset_split.py
```

---

## Step 3: Fine-Tune the Model

Submit the SLURM job:

```bash
mkdir -p logs
sbatch finetune_slurm.sh
```

Monitor the job:

```bash
squeue -u $USER
tail -f logs/finetune_<JOB_ID>.out
```

The fine-tuned model will be saved to `./results/`.

---

## Step 4: Evaluate the Fine-Tuned Model

```bash
python3 basic_inference.py
```

Custom prompt:

```bash
python3 basic_inference.py \
  --model ./results \
  --message "What are the early symptoms of diabetes?"
```

---

## Project Structure

```
llm_finetuning/
├── prepare_dataset_split.py   # Step 2: download + split dataset
├── finetune_model.py          # Step 3: training script
├── finetune_slurm.sh          # Step 3: SLURM job script
├── ds_config.json             # DeepSpeed ZeRO Stage 2 config
├── basic_inference.py         # Step 4: evaluate fine-tuned model
├── requirements.txt
└── data/                      # dataset cache + splits (gitignored)
    └── medical_dataset_split/
```
