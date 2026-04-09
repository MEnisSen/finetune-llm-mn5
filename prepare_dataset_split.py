import os

from datasets import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLIT_DIR = os.path.join(DATA_DIR, "medical_dataset_split")
DATASET_NAME = "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B"
TEST_SIZE = 0.1
SEED = 42


def format_conversation(example: dict) -> dict:
    conversation = ""
    for msg in example["messages"]:
        conversation += f"{msg['role']}: {msg['content']}\n"
    return {"text": conversation.strip()}


def prepare_dataset() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, cache_dir=DATA_DIR)

    print("Formatting conversations...")
    dataset = dataset.map(format_conversation, remove_columns=dataset["train"].column_names)

    print(f"Creating train/test split (test_size={TEST_SIZE})...")
    split_dataset = dataset["train"].train_test_split(test_size=TEST_SIZE, seed=SEED)

    print(f"Saving split dataset to: {SPLIT_DIR}")
    split_dataset.save_to_disk(SPLIT_DIR)

    print("\nDataset split complete:")
    print(f"  Train examples : {len(split_dataset['train'])}")
    print(f"  Test examples  : {len(split_dataset['test'])}")
    print(f"  Saved to       : {SPLIT_DIR}")


if __name__ == "__main__":
    prepare_dataset()
