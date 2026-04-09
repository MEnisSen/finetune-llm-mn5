import os

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

# Load dataset and model
DATA_DIR = "./data"
MODELS_DIR = "./models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

dataset = load_dataset("OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B", cache_dir=DATA_DIR)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", cache_dir=MODELS_DIR)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", cache_dir=MODELS_DIR)

# Prepare data for training
def format_conversation(example):
    conversation = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        conversation += f"{role}: {content}\n"
    return {"text": conversation}

formatted_dataset = dataset.map(format_conversation)

# Training setup
training_args = TrainingArguments(
    output_dir="./models/medical-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    save_steps=500,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
)

trainer.train()