"""
benchmark_latency.py

Sends a batch of requests to a running serve.py instance and measures
end-to-end latency. Run this once against the merged (unquantized) model
and once against the quantized model to produce the "latency before/after
quantization" comparison referenced in the project write-up.

Usage:
    python src/benchmark_latency.py --url http://localhost:8000/chat_sync --label merged
    python src/benchmark_latency.py --url http://localhost:8000/chat_sync --label quantized
"""
import argparse
import json
import statistics
import time

import requests

SAMPLE_QUERIES = [
    "My order hasn't arrived yet, what should I do?",
    "How do I request a refund for a damaged item?",
    "Can I change the shipping address after placing an order?",
    "The item I received doesn't match the description, what are my options?",
    "How long does standard shipping usually take?",
    "I want to cancel my subscription, how do I do that?",
    "Do you offer international shipping?",
    "My discount code isn't working at checkout.",
    "How can I track my package?",
    "I was charged twice for the same order.",
]


def run_benchmark(url, n_requests, max_tokens):
    latencies = []
    for i in range(n_requests):
        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        start = time.time()
        resp = requests.post(url, json={"message": query, "max_tokens": max_tokens})
        resp.raise_for_status()
        latencies.append(time.time() - start)
    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--label", required=True,
        help="e.g. 'merged' or 'quantized', used in the results file name",
    )
    parser.add_argument("--n_requests", type=int, default=30)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    print(f"Benchmarking {args.url} ({args.n_requests} requests)...")
    latencies = run_benchmark(args.url, args.n_requests, args.max_tokens)

    summary = {
        "label": args.label,
        "n_requests": args.n_requests,
        "mean_latency_s": round(statistics.mean(latencies), 3),
        "p50_latency_s": round(statistics.median(latencies), 3),
        "p95_latency_s": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 3),
        "min_latency_s": round(min(latencies), 3),
        "max_latency_s": round(max(latencies), 3),
    }

    out_path = f"{args.output_dir}/latency_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
