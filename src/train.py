"""
train.py

LoRA (default) or QLoRA (--use_qlora) instruction fine-tuning of a 7-8B
base model on the customer-support dataset built by prepare_data.py.

On a single 48GB GPU, standard bf16 LoRA is comfortable for a 7-8B model --
QLoRA is provided as an optional ablation (--use_qlora) to compare memory
footprint and training speed against standard LoRA, not because it's
required at this model size.

Usage:
    python src/train.py --config config.yaml
    python src/train.py --config config.yaml --use_qlora
"""
import argparse
import json
import time

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def formatting_func(tokenizer):
    def _format(example):
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return _format


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--use_qlora", action="store_true",
        help="4-bit QLoRA instead of standard bf16 LoRA",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    max_seq_len = cfg["model"]["max_seq_len"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["train"]
    data_dir = cfg["paths"]["data_dir"]
    output_dir = cfg["paths"]["output_dir"] + ("-qlora" if args.use_qlora else "-lora")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False  # required for gradient checkpointing

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=f"{data_dir}/train.jsonl", split="train")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        # warmup_ratio was merged into warmup_steps in this transformers version:
        # a float in [0, 1) is now interpreted as a ratio of total steps.
        warmup_steps=train_cfg["warmup_ratio"],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func(tokenizer),
        peft_config=peft_config,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    run_summary = {
        "mode": "qlora" if args.use_qlora else "lora",
        "model_name": model_name,
        "output_dir": output_dir,
        "training_time_seconds": round(elapsed, 1),
        "peak_gpu_memory_gb": round(peak_mem_gb, 2),
    }
    with open(f"{output_dir}/run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Training complete. Adapter saved to {output_dir}")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
