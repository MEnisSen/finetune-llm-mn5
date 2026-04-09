import os

from datasets import DatasetDict, concatenate_datasets, load_dataset


def prepare_split(
    dataset_name: str,
    output_dir: str = "./data/medical_dataset_split",
    cache_dir: str = "./data",
    test_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Load a Hugging Face dataset, shuffle it, and create train/test splits.

    Args:
        dataset_name: Hugging Face dataset ID.
        output_dir: Local directory to save split dataset.
        cache_dir: Local directory to cache Hugging Face dataset downloads.
        test_size: Fraction of examples reserved for test.
        seed: Random seed for deterministic shuffle/split.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    # If the dataset already has a train split, split from that.
    # Otherwise, combine all available splits so we can define our own holdout.
    if "train" in dataset:
        source = dataset["train"]
    else:
        source = concatenate_datasets([split for split in dataset.values()])

    shuffled = source.shuffle(seed=seed)
    split_dataset = shuffled.train_test_split(test_size=test_size, seed=seed)

    split_dataset.save_to_disk(output_dir)
    return split_dataset


if __name__ == "__main__":
    DATASET_NAME = "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B"
    OUTPUT_DIR = "./data/medical_dataset_split"
    CACHE_DIR = "./data"
    TEST_SIZE = 0.1
    SEED = 42

    split_dataset = prepare_split(
        dataset_name=DATASET_NAME,
        output_dir=OUTPUT_DIR,
        cache_dir=CACHE_DIR,
        test_size=TEST_SIZE,
        seed=SEED,
    )

    print("Saved dataset split:")
    print(split_dataset)
    print(f"\nTrain examples: {len(split_dataset['train'])}")
    print(f"Test examples:  {len(split_dataset['test'])}")
    print(f"Saved to: {OUTPUT_DIR}")
