"""
prepare_data.py

Downloads the Bitext Customer Support instruction dataset, formats it into
chat-style instruction/response pairs, and creates:
  1. A train split for fine-tuning
  2. A held-out domain test split (for before/after quality comparison)
  3. A general-capability probe set (MMLU subset) used to check for
     catastrophic forgetting after fine-tuning

Usage:
    python src/prepare_data.py --config config.yaml
"""
import argparse
import json
import os
import random

import yaml
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a helpful, professional customer support assistant. "
    "Answer the customer's question clearly and concisely."
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def format_domain_example(instruction, response):
    return {
        "system": SYSTEM_PROMPT,
        "instruction": instruction.strip(),
        "response": response.strip(),
    }


def build_domain_splits(cfg):
    dataset_id = cfg["data"]["dataset_id"]
    test_size = cfg["data"]["test_size"]
    seed = cfg["data"]["seed"]

    print(f"Loading {dataset_id} from the Hugging Face Hub...")
    ds = load_dataset(dataset_id, split="train")

    # The Bitext dataset ships 'instruction' and 'response' columns.
    # Guard against schema drift so the script fails loudly, not silently.
    required_cols = {"instruction", "response"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Expected columns {required_cols} not found in dataset. "
            f"Available columns: {ds.column_names}. "
            f"Update format_domain_example() to match the actual schema."
        )

    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=test_size, seed=seed)
    train_ds, test_ds = split["train"], split["test"]

    train_examples = [
        format_domain_example(r["instruction"], r["response"]) for r in train_ds
    ]
    test_examples = [
        format_domain_example(r["instruction"], r["response"]) for r in test_ds
    ]

    print(f"Domain train examples: {len(train_examples)}")
    print(f"Domain test examples:  {len(test_examples)}")
    return train_examples, test_examples


def build_general_probe(cfg):
    """
    Builds a small, stratified MMLU subset used only to check whether
    fine-tuning degraded the model's general knowledge/reasoning
    (i.e. a catastrophic-forgetting probe). Not used for training.
    """
    probe_size = cfg["data"]["general_probe_size"]
    seed = cfg["data"]["seed"]
    random.seed(seed)

    print("Loading MMLU (general capability probe)...")
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception as e:
        print(f"Falling back to a fixed subject list ({e})")
        subjects = [
            "elementary_mathematics",
            "high_school_psychology",
            "professional_law",
            "marketing",
            "college_computer_science",
        ]
        from datasets import concatenate_datasets

        parts = [load_dataset("cais/mmlu", s, split="test") for s in subjects]
        ds = concatenate_datasets(parts)

    indices = list(range(len(ds)))
    random.shuffle(indices)
    subset = ds.select(indices[:probe_size])

    letters = ["A", "B", "C", "D"]
    probe_examples = []
    for row in subset:
        choices_fmt = "\n".join(
            f"{letters[i]}. {c}" for i, c in enumerate(row["choices"])
        )
        prompt = (
            f"{row['question']}\n\n{choices_fmt}\n\n"
            "Answer with only the letter of the correct choice."
        )
        probe_examples.append({"prompt": prompt, "answer": letters[row["answer"]]})

    print(f"General capability probe examples: {len(probe_examples)}")
    return probe_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg["paths"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    train_examples, test_examples = build_domain_splits(cfg)
    general_examples = build_general_probe(cfg)

    def dump(path, rows):
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    dump(os.path.join(data_dir, "train.jsonl"), train_examples)
    dump(os.path.join(data_dir, "domain_test.jsonl"), test_examples)
    dump(os.path.join(data_dir, "general_probe.jsonl"), general_examples)

    print(f"Wrote data files to {data_dir}/")


if __name__ == "__main__":
    main()
