import os

import torch
from datasets import load_from_disk
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
DS_CONFIG = os.path.join(PROJECT_ROOT, "ds_config.json")
MAX_LENGTH = 512


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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=os.path.join(PROJECT_ROOT, "logs"),
        logging_steps=100,
        deepspeed=DS_CONFIG,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nFine-tuned model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
