import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

DATA_DIR = "./data"
MODELS_DIR = "./models"
MODEL_ID = "Qwen/Qwen3.5-4B"
MAX_LENGTH = 1024

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODELS_DIR)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=MODELS_DIR)
model.gradient_checkpointing_enable()

dataset = load_dataset("OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B", cache_dir=DATA_DIR)


def format_and_tokenize(example):
    conversation = ""
    for msg in example["messages"]:
        conversation += f"{msg['role']}: {msg['content']}\n"
    return tokenizer(
        conversation,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


tokenized_dataset = dataset.map(
    format_and_tokenize,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./models/medical-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()
