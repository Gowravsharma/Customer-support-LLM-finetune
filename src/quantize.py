"""
quantize.py

Two-step production prep:
  1. Merge the trained LoRA adapter into the base model (a merged model
     is what you actually deploy -- you don't serve LoRA deltas separately
     unless you specifically need multi-adapter serving).
  2. AWQ-quantize the merged model to 4-bit for efficient serving.

If autoawq isn't installed or isn't compatible with your CUDA build, the
script still leaves you a working merged bf16 model that serve.py can load
directly (with bitsandbytes 4-bit at load time as a lighter-weight
fallback quantization path).

Usage:
    python src/quantize.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --adapter_path checkpoints/lora-customer-support-lora \
        --merged_dir checkpoints/merged \
        --quantized_dir checkpoints/quantized
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter(base_model, adapter_path, merged_dir):
    print(f"Loading base model {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Attaching adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")
    return merged_dir


def quantize_awq(merged_dir, quantized_dir):
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer as _Tok
    except ImportError:
        print(
            "autoawq is not installed or failed to import. Skipping AWQ export.\n"
            "You can still serve the merged model directly with bitsandbytes "
            "4-bit loading in serve.py (set QUANTIZE_AT_LOAD=1)."
        )
        return None

    print(f"Loading merged model from {merged_dir} for AWQ quantization...")
    tokenizer = _Tok.from_pretrained(merged_dir)
    model = AutoAWQForCausalLM.from_pretrained(merged_dir)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    print("Running AWQ quantization (this can take a while)...")
    model.quantize(tokenizer, quant_config=quant_config)

    os.makedirs(quantized_dir, exist_ok=True)
    model.save_quantized(quantized_dir)
    tokenizer.save_pretrained(quantized_dir)
    print(f"Quantized model saved to {quantized_dir}")
    return quantized_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--merged_dir", default="checkpoints/merged")
    parser.add_argument("--quantized_dir", default="checkpoints/quantized")
    parser.add_argument(
        "--skip_quantize", action="store_true", help="Only merge, skip AWQ step"
    )
    args = parser.parse_args()

    merge_adapter(args.base_model, args.adapter_path, args.merged_dir)

    if not args.skip_quantize:
        quantize_awq(args.merged_dir, args.quantized_dir)


if __name__ == "__main__":
    main()
