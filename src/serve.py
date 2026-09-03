"""
serve.py

FastAPI wrapper around vLLM serving the fine-tuned (merged, optionally
quantized) customer-support model, with SSE streaming -- mirroring the
Financial Query project's serving style.

Usage:
    MODEL_PATH=checkpoints/merged python src/serve.py
    # or, to serve the AWQ-quantized model:
    MODEL_PATH=checkpoints/quantized VLLM_QUANTIZATION=awq python src/serve.py

Then:
    curl -N -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "My order has not arrived yet, what do I do?"}'
"""
import os
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from vllm import LLM, SamplingParams

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/merged")
QUANTIZATION = os.environ.get("VLLM_QUANTIZATION")  # e.g. "awq", or unset

SYSTEM_PROMPT = (
    "You are a helpful, professional customer support assistant. "
    "Answer the customer's question clearly and concisely."
)

app = FastAPI(title="Customer Support LLM API")

# AWQ kernels only support float16, not bfloat16.
DTYPE = "float16" if QUANTIZATION == "awq" else "auto"

print(f"Loading model from {MODEL_PATH} (quantization={QUANTIZATION}, dtype={DTYPE})...")
llm = LLM(model=MODEL_PATH, quantization=QUANTIZATION, dtype=DTYPE)


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 256
    temperature: float = 0.2


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/chat_sync")
def chat_sync(req: ChatRequest):
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{req.message}\n<|assistant|>\n"
    params = SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature)
    start = time.time()
    output = llm.generate([prompt], params)[0]
    latency = time.time() - start
    return {
        "response": output.outputs[0].text.strip(),
        "latency_seconds": round(latency, 3),
    }


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{req.message}\n<|assistant|>\n"
    params = SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature)

    async def event_generator():
        start = time.time()
        # NOTE: this calls the synchronous llm.generate() and then streams the
        # already-complete text word-by-word for a simple, readable demo. For
        # true token-by-token streaming under concurrent load, swap to
        # vllm.AsyncLLMEngine, which exposes a native async generator.
        output = llm.generate([prompt], params)[0]
        text = output.outputs[0].text.strip()
        first_token_time = time.time()
        for chunk in text.split(" "):
            yield {"data": chunk + " "}
        total_latency = time.time() - start
        yield {
            "event": "done",
            "data": (
                f"ttft_seconds={round(first_token_time - start, 3)} "
                f"total_seconds={round(total_latency, 3)}"
            ),
        }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
