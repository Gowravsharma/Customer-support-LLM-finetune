# Customer Support LLM — Domain Fine-Tuning, Deployment & Evaluation

A fine-tuned, quantized, and served 7–8B open-source LLM specialized for
customer-support response generation — with a measured before/after
comparison on both domain quality and general capability retention.

## Why this project exists

Most portfolio LLM projects stop at "I called an API" or "I ran a LoRA
script in a notebook." This one is built to prove three things end to end:

1. **Fine-tuning actually works** — measured ROUGE-L improvement on
   held-out customer-support queries, base model vs. fine-tuned model.
2. **The model didn't forget everything else** — the same fine-tuned
   model is also scored on an MMLU probe set, so the "catastrophic
   forgetting" question has a real number attached to it, not just a claim.
3. **It's actually deployable** — the adapter is merged, quantized (AWQ),
   and served behind a FastAPI endpoint with streaming, with latency
   measured before and after quantization.

## Pipeline

```
prepare_data.py  ->  run_eval.py (baseline)  ->  train.py  ->  run_eval.py (post-train)
                                                       |
                                               quantize.py (merge + AWQ)
                                                       |
                                                 serve.py (FastAPI + vLLM)
                                                       |
                                             benchmark_latency.py
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # needed to pull the base model + datasets
```

Hardware used: single NVIDIA L40, 48GB VRAM.

## Usage

```bash
# 1. Build train/test splits + the general-capability probe set
python src/prepare_data.py --config config.yaml

# 2. Baseline eval (BEFORE fine-tuning) -- run this first, don't skip it
python src/run_eval.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --output_file results/baseline.json

# 3. Fine-tune (standard bf16 LoRA; add --use_qlora for the QLoRA ablation)
python src/train.py --config config.yaml
python src/train.py --config config.yaml --use_qlora   # optional comparison run

# 4. Post-fine-tune eval, same test sets as step 2
python src/run_eval.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_path checkpoints/lora-customer-support-lora \
    --output_file results/finetuned.json

# 5. Merge + quantize for deployment
python src/quantize.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_path checkpoints/lora-customer-support-lora \
    --merged_dir checkpoints/merged \
    --quantized_dir checkpoints/quantized

# 6. Serve (run once per variant to benchmark both)
MODEL_PATH=checkpoints/merged python src/serve.py                                        # terminal A
python src/benchmark_latency.py --url http://localhost:8000/chat_sync --label merged     # terminal B

MODEL_PATH=checkpoints/quantized VLLM_QUANTIZATION=awq python src/serve.py               # terminal A
python src/benchmark_latency.py --url http://localhost:8000/chat_sync --label quantized  # terminal B
```

## Design notes

- **Dataset**: [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
  (~27k examples, 27 intents) — a real, purpose-built instruction dataset
  for this exact use case, not a synthetic toy set.
- **Why standard LoRA, not QLoRA, by default**: on a 48GB card, a 7–8B
  model fine-tunes comfortably in bf16 without needing 4-bit training
  tricks. QLoRA is included as an optional `--use_qlora` run specifically
  to *measure* the memory/speed tradeoff, rather than assuming it's needed.
- **Why a general-capability probe**: LoRA freezes the base weights, which
  should limit catastrophic forgetting in theory — this project checks
  that in practice, on a held-out MMLU subset, rather than treating it
  as a given.
- **Why AWQ for deployment**: AWQ quantization is directly compatible
  with vLLM serving and is a common production choice; a bitsandbytes
  4-bit load-time fallback is documented in `quantize.py` if AWQ isn't
  available in your environment.
- **`trl` version note**: `SFTTrainer`'s `formatting_func` signature has
  changed across `trl` releases. The version used in `train.py` (per-example
  formatting) matches recent stable releases at time of writing — check
  your installed version's docs if you hit a signature error.

## Results

*(Fill in after running the pipeline above — see `results/RESULTS_TEMPLATE.md`
for the exact table to complete.)*

## Acknowledgements

- Base model: Qwen2.5-7B-Instruct (Apache 2.0) — swap in `config.yaml`
  for Llama-3.1-8B-Instruct or another comparable model if preferred.
- Dataset: Bitext Customer Support LLM Chatbot Training Dataset (Bitext, CC BY 4.0).
