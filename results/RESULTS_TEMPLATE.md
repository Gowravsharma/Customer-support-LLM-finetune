# Results (fill in after running the pipeline)

## Domain quality (ROUGE-L, higher is better)

| | Base model | Fine-tuned model | Δ |
|---|---|---|---|
| ROUGE-L | 23.66 | 43.47 | +19.81 (+83.7%) |

## General capability retention (MMLU accuracy, higher is better)

| | Base model | Fine-tuned model | Δ |
|---|---|---|---|
| MMLU accuracy | 68.33% | 71.67% | +3.34 pp |

> A small negative Δ here is expected and fine — the goal is to confirm
> it's small, not zero. A large drop would indicate real forgetting.
> (Here Δ is actually positive — no forgetting signal at all, on a 60-example probe.)

## Deployment latency (mean, seconds)

| | Merged (bf16) | Quantized (AWQ, generic kernel) | Quantized (AWQ-Marlin kernel) |
|---|---|---|---|
| Mean latency | 2.601s | 6.193s | **0.953s** |
| P95 latency | 2.654s | 6.561s | **1.014s** |

> vLLM's generic AWQ GEMM kernel was *slower* than the unquantized model
> (dequantization overhead not amortized at batch size 1). Switching to the
> Marlin-optimized AWQ kernel (`quantization=awq_marlin`, vLLM auto-detects
> it's available and recommends it) reversed this completely: **2.7x faster
> than the unquantized bf16 model**, and 6.5x faster than the generic AWQ
> kernel, while keeping AWQ's ~2.7x memory reduction (14.2GB -> 5.2GB).
> Lesson: quantization kernel choice matters as much as quantization itself.

## Training run details

| | Standard LoRA | QLoRA (ablation) |
|---|---|---|
| Peak GPU memory (GB) | 19.84 | not run |
| Training time | 9640.2s (~2h 40m) | not run |

---

Once filled in, turn this into 3 resume bullets:
1. Domain quality improvement (the headline number)
2. General capability retention (your forgetting-mitigation evidence)
3. Deployment latency after quantization (your production-readiness proof)
