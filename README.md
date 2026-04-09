# LLM Fine-Tuning (Basic)

Minimal scripts for dataset prep, training, and quick inference checks.

## Setup

```bash
pip3 install -r requirements.txt
```

## Run scripts

### 1) Prepare dataset split

Creates train/validation split files from your dataset script.

```bash
python3 prepare_dataset_split.py
```

### 2) Fine-tune model

Starts training using settings in `finetune_model.py`.

```bash
python3 finetune_model.py
```

### 3) Basic inference test

Runs a simple prompt -> response test with a model.

```bash
python3 basic_inference.py
```

Optional custom prompt/model:

```bash
python3 basic_inference.py --model "Qwen/Qwen3.5-4B" --message "Hello, can you explain fine-tuning in one sentence?"
```
