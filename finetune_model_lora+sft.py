import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.path.join(PROJECT_ROOT, "models", "Qwen3.5-4B")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "medical_dataset_split")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_lora_sft")
DS_CONFIG = os.path.join(PROJECT_ROOT, "ds_config.json")
MAX_LENGTH = 512

# ---------------------------------------------------------------------------
# LoRA config
# With 4x H100s you have plenty of VRAM — no need for 4-bit quantisation.
# Plain bf16 LoRA works cleanly with torchrun + DeepSpeed DDP.
# ---------------------------------------------------------------------------
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def tokenize(example: dict, tokenizer) -> dict:
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset split from disk...")
    dataset = load_from_disk(SPLIT_DIR)

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    # ------------------------------------------------------------------
    # Load in bf16 — no device_map="auto", torchrun handles device placement
    # ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Inject LoRA adapters — only adapter weights will be trained
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()  # expect ~0.5–2% of total params

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,   # effective batch = 2 * 4 GPUs * 8 = 64
        num_train_epochs=3,
        learning_rate=2e-4,              # LoRA works better at higher LR than full FT
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=os.path.join(PROJECT_ROOT, "logs"),
        logging_steps=100,
        deepspeed=DS_CONFIG,             # ZeRO still helps with optimizer state sharding
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # Saves only the LoRA adapter weights (a few hundred MB, not the full model)
    trainer.save_model(OUTPUT_DIR)
    print(f"\nLoRA adapter saved to: {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # Optional: merge adapter into base model for a standalone checkpoint
    # ------------------------------------------------------------------
    # from peft import PeftModel
    # base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    # merged = PeftModel.from_pretrained(base, OUTPUT_DIR).merge_and_unload()
    # merged.save_pretrained(os.path.join(OUTPUT_DIR, "merged"))
    # tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "merged"))


if __name__ == "__main__":
    main()