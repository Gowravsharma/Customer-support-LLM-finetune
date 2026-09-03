"""
eval_utils.py

Shared helpers for scoring model outputs:
  - Domain response quality: ROUGE-L against reference support replies
  - General capability: multiple-choice accuracy on the MMLU probe

Kept separate from run_eval.py so both the baseline and post-fine-tune
evaluation runs share exactly the same scoring logic -- otherwise a
"before vs after" comparison isn't trustworthy.
"""
import re

import torch
import evaluate

_rouge = evaluate.load("rouge")


def build_chat_prompt(tokenizer, system_prompt, user_instruction):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instruction},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens=256, batch_size=8):
    """Greedy-decodes a list of already-formatted prompts in batches."""
    model.eval()
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for j in range(len(batch)):
            input_len = inputs["input_ids"][j].shape[0]
            text = tokenizer.decode(gen[j][input_len:], skip_special_tokens=True)
            outputs.append(text.strip())
    return outputs


def compute_rouge_l(predictions, references):
    result = _rouge.compute(
        predictions=predictions, references=references, rouge_types=["rougeL"]
    )
    return round(result["rougeL"] * 100, 2)


def extract_letter(text):
    """Pulls a single A/B/C/D choice out of a (possibly noisy) generation."""
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    return match.group(1) if match else None


def compute_mmlu_accuracy(predictions, gold_answers):
    correct = 0
    scored = 0
    for pred, gold in zip(predictions, gold_answers):
        letter = extract_letter(pred)
        if letter is None:
            continue
        scored += 1
        if letter == gold:
            correct += 1
    accuracy = round(100 * correct / len(gold_answers), 2) if gold_answers else 0.0
    return {
        "accuracy_pct": accuracy,
        "answered": scored,
        "total": len(gold_answers),
    }
