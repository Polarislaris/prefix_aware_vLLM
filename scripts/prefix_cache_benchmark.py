#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from openai import OpenAI


@dataclass
class RequestResult:
    mode: str
    index: int
    prompt_id: str
    latency_s: float
    prompt_chars: int
    output_tokens: int


def _id_sort_key(item: dict) -> int:
    raw_id = str(item.get("id", ""))
    digits = "".join(ch for ch in raw_id if ch.isdigit())
    return int(digits) if digits else 0


def load_prompts(prompt_file: Path, requests_per_mode: int) -> tuple[list[dict], list[dict], dict]:
    data = json.loads(prompt_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Prompt file must be a JSON list: {prompt_file}")

    shared_items = [x for x in data if isinstance(x, dict) and x.get("category") == "shared"]
    unique_items = [x for x in data if isinstance(x, dict) and x.get("category") == "unique"]

    shared_items.sort(key=_id_sort_key)
    unique_items.sort(key=_id_sort_key)

    if len(shared_items) < requests_per_mode:
        raise ValueError(
            f"Not enough shared prompts. Need {requests_per_mode}, got {len(shared_items)}"
        )
    if len(unique_items) < requests_per_mode:
        raise ValueError(
            f"Not enough unique prompts. Need {requests_per_mode}, got {len(unique_items)}"
        )

    selected_shared = shared_items[:requests_per_mode]
    selected_unique = unique_items[:requests_per_mode]

    meta = {
        "prompt_file": str(prompt_file),
        "dataset_total": len(data),
        "dataset_shared": len(shared_items),
        "dataset_unique": len(unique_items),
        "selected_per_mode": requests_per_mode,
        "selected_shared_ids": [x.get("id", "") for x in selected_shared],
        "selected_unique_ids": [x.get("id", "") for x in selected_unique],
    }
    return selected_shared, selected_unique, meta


def run_once(client: OpenAI, model: str, prompt: str, max_tokens: int, temperature: float) -> tuple[float, int]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    t1 = time.perf_counter()

    out_tokens = 0
    if resp.usage is not None and resp.usage.completion_tokens is not None:
        out_tokens = int(resp.usage.completion_tokens)

    return t1 - t0, out_tokens


def summarize(results: list[RequestResult]) -> dict:
    lats = [r.latency_s for r in results]
    tps = [
        (r.output_tokens / r.latency_s) if r.latency_s > 0 and r.output_tokens > 0 else 0.0
        for r in results
    ]

    summary = {
        "count": len(results),
        "latency_mean_s": statistics.mean(lats) if lats else 0.0,
        "latency_p50_s": statistics.median(lats) if lats else 0.0,
        "latency_p90_s": statistics.quantiles(lats, n=10)[8] if len(lats) >= 10 else max(lats, default=0.0),
        "latency_min_s": min(lats, default=0.0),
        "latency_max_s": max(lats, default=0.0),
        "throughput_tokens_per_s_mean": statistics.mean(tps) if tps else 0.0,
    }
    return summary


def warmup(client: OpenAI, model: str, prompt: str, max_tokens: int, temperature: float, n: int) -> None:
    for _ in range(n):
        run_once(client, model, prompt, max_tokens=max_tokens, temperature=temperature)


def write_request_csv(results: list[RequestResult], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode",
            "index",
            "prompt_id",
            "latency_s",
            "prompt_chars",
            "output_tokens",
            "throughput_tokens_per_s",
        ])
        for r in results:
            throughput = (r.output_tokens / r.latency_s) if r.latency_s > 0 and r.output_tokens > 0 else 0.0
            writer.writerow([
                r.mode,
                r.index,
                r.prompt_id,
                f"{r.latency_s:.6f}",
                r.prompt_chars,
                r.output_tokens,
                f"{throughput:.6f}",
            ])


def write_summary_csv(shared_summary: dict, unique_summary: dict, speedup: dict, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group",
            "count",
            "latency_mean_s",
            "latency_p50_s",
            "latency_p90_s",
            "latency_min_s",
            "latency_max_s",
            "throughput_tokens_per_s_mean",
        ])
        writer.writerow([
            "shared_prefix",
            shared_summary["count"],
            f"{shared_summary['latency_mean_s']:.6f}",
            f"{shared_summary['latency_p50_s']:.6f}",
            f"{shared_summary['latency_p90_s']:.6f}",
            f"{shared_summary['latency_min_s']:.6f}",
            f"{shared_summary['latency_max_s']:.6f}",
            f"{shared_summary['throughput_tokens_per_s_mean']:.6f}",
        ])
        writer.writerow([
            "unique_prefix",
            unique_summary["count"],
            f"{unique_summary['latency_mean_s']:.6f}",
            f"{unique_summary['latency_p50_s']:.6f}",
            f"{unique_summary['latency_p90_s']:.6f}",
            f"{unique_summary['latency_min_s']:.6f}",
            f"{unique_summary['latency_max_s']:.6f}",
            f"{unique_summary['throughput_tokens_per_s_mean']:.6f}",
        ])
        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerow(["latency_mean_speedup_x", f"{speedup['latency_mean_speedup_x']:.6f}"])
        writer.writerow(["latency_p50_speedup_x", f"{speedup['latency_p50_speedup_x']:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vLLM prefix caching with shared vs unique prefixes")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="testkey", help="API key for vLLM")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--requests", type=int, default=50, help="Requests per mode")
    parser.add_argument(
        "--prompt-file",
        default="/home/xy68/kvcache_lab/vllm_shared_prefix_prompts_100.json",
        help="JSON prompt file containing category=shared|unique and prompt fields",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests")
    parser.add_argument("--output-dir", default="/home/xy68/kvcache_lab/data", help="Directory to save results")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    shared_prompts, unique_prompts, prompt_meta = load_prompts(
        prompt_file=Path(args.prompt_file),
        requests_per_mode=args.requests,
    )

    if args.warmup > 0:
        warmup(
            client,
            args.model,
            shared_prompts[0]["prompt"],
            max_tokens=min(args.max_tokens, 64),
            temperature=args.temperature,
            n=args.warmup,
        )

    all_results: list[RequestResult] = []

    print(f"Running SHARED_PREFIX mode ({args.requests} requests)...")
    for i, item in enumerate(shared_prompts):
        prompt = str(item["prompt"])
        lat, out_tokens = run_once(
            client,
            args.model,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        all_results.append(
            RequestResult(
                mode="shared_prefix",
                index=i,
                prompt_id=str(item.get("id", "")),
                latency_s=lat,
                prompt_chars=len(prompt),
                output_tokens=out_tokens,
            )
        )
        print(f"  shared #{i:02d}: {lat:.3f}s, out_tokens={out_tokens}")

    print(f"Running UNIQUE_PREFIX mode ({args.requests} requests)...")
    for i, item in enumerate(unique_prompts):
        prompt = str(item["prompt"])
        lat, out_tokens = run_once(
            client,
            args.model,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        all_results.append(
            RequestResult(
                mode="unique_prefix",
                index=i,
                prompt_id=str(item.get("id", "")),
                latency_s=lat,
                prompt_chars=len(prompt),
                output_tokens=out_tokens,
            )
        )
        print(f"  unique #{i:02d}: {lat:.3f}s, out_tokens={out_tokens}")

    shared = [r for r in all_results if r.mode == "shared_prefix"]
    unique = [r for r in all_results if r.mode == "unique_prefix"]

    shared_summary = summarize(shared)
    unique_summary = summarize(unique)

    speedup = {
        "latency_mean_speedup_x": (
            unique_summary["latency_mean_s"] / shared_summary["latency_mean_s"]
            if shared_summary["latency_mean_s"] > 0
            else 0.0
        ),
        "latency_p50_speedup_x": (
            unique_summary["latency_p50_s"] / shared_summary["latency_p50_s"]
            if shared_summary["latency_p50_s"] > 0
            else 0.0
        ),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    request_csv_file = out_dir / f"prefix_cache_requests_{ts}.csv"
    summary_csv_file = out_dir / f"prefix_cache_summary_{ts}.csv"
    meta_json_file = out_dir / f"prefix_cache_meta_{ts}.json"

    write_request_csv(all_results, request_csv_file)
    write_summary_csv(shared_summary, unique_summary, speedup, summary_csv_file)
    meta_json_file.write_text(
        json.dumps(
            {
                "config": vars(args),
                "prompt_selection": prompt_meta,
                "results_preview": [asdict(r) for r in all_results[:5]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    print("Shared prefix:", json.dumps(shared_summary, indent=2))
    print("Unique prefix:", json.dumps(unique_summary, indent=2))
    print("Speedup:", json.dumps(speedup, indent=2))
    print(f"\nSaved request CSV to: {request_csv_file}")
    print(f"Saved summary CSV to: {summary_csv_file}")
    print(f"Saved meta JSON to: {meta_json_file}")


if __name__ == "__main__":
    main()
