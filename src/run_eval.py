"""
run_eval.py

Runs a model (base OR fine-tuned, merged OR base+adapter) against:
  1. The domain test set (customer-support ROUGE-L quality)
  2. The general-capability probe (MMLU accuracy -- forgetting check)

Used TWICE in this project:
  - Once before fine-tuning, to record the baseline
  - Once after fine-tuning, to measure the delta

Usage:
    python src/run_eval.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --output_file results/baseline.json

    python src/run_eval.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --adapter_path checkpoints/lora-customer-support-lora \
        --output_file results/finetuned.json
"""
import argparse
import json
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from eval_utils import (
    build_chat_prompt,
    generate_batch,
    compute_rouge_l,
    compute_mmlu_accuracy,
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_model(model_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_path:
        print(f"Attaching LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def run_domain_eval(model, tokenizer, domain_examples, max_new_tokens):
    prompts = [
        build_chat_prompt(tokenizer, ex["system"], ex["instruction"])
        for ex in domain_examples
    ]
    references = [ex["response"] for ex in domain_examples]
    predictions = generate_batch(
        model, tokenizer, prompts, max_new_tokens=max_new_tokens
    )
    rouge_l = compute_rouge_l(predictions, references)
    return rouge_l, predictions


def run_general_eval(model, tokenizer, general_examples, max_new_tokens=8):
    prompts = [
        build_chat_prompt(
            tokenizer,
            "Answer the multiple-choice question with a single letter.",
            ex["prompt"],
        )
        for ex in general_examples
    ]
    gold = [ex["answer"] for ex in general_examples]
    predictions = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
    return compute_mmlu_accuracy(predictions, gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", required=True,
        help="Base model id/path, or merged fine-tuned model path",
    )
    parser.add_argument(
        "--adapter_path", default=None,
        help="Optional LoRA adapter path, if not yet merged into model_path",
    )
    parser.add_argument("--domain_test_file", default="data/domain_test.jsonl")
    parser.add_argument("--general_test_file", default="data/general_probe.jsonl")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    domain_examples = load_jsonl(args.domain_test_file)
    general_examples = load_jsonl(args.general_test_file)

    model, tokenizer = load_model(args.model_path, args.adapter_path)

    print("Running domain (customer support) evaluation...")
    rouge_l, domain_preds = run_domain_eval(
        model, tokenizer, domain_examples, args.max_new_tokens
    )
    print(f"  ROUGE-L: {rouge_l}")

    print("Running general-capability (MMLU) evaluation...")
    general_result = run_general_eval(model, tokenizer, general_examples)
    print(f"  MMLU accuracy: {general_result['accuracy_pct']}%")

    results = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domain_rouge_l": rouge_l,
        "general_mmlu": general_result,
        "num_domain_examples": len(domain_examples),
        "num_general_examples": len(general_examples),
    }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
