#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class PromptItem:
    req_id: int
    prompt: str
    prefix_type: str


@dataclass
class Scenario:
    name: str
    route: str
    grouping: str
    window_ms: int


@dataclass
class RequestResult:
    scenario: str
    route: str
    grouping: str
    window_ms: int
    order_index: int
    req_id: int
    prefix_type: str
    group_key: str
    schedule_wait_ms: float
    prompt_tokens: int
    prefix_tokens: int
    reused_tokens: int
    reuse_hit: int
    model_latency_s: float
    total_latency_s: float
    output_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prefix-similarity aware benchmark: route1 grouping, route2 scheduling window, "
            "route3 cache effectiveness metrics"
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="testkey", help="API key for vLLM")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument(
        "--prompt-file",
        default="/home/xy68/kvcache_lab/vllm_shared_prefix_10types_100.json",
        help="Input JSON prompt file",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="How many prompts to benchmark")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests before benchmark")
    parser.add_argument("--exact-prefix-chars", type=int, default=120, help="Chars used by exact grouping")
    parser.add_argument("--sim-prefix-tokens", type=int, default=24, help="Tokens used by similarity grouping")
    parser.add_argument(
        "--sim-threshold-bits",
        type=int,
        default=8,
        help="SimHash hamming threshold for clustering",
    )
    parser.add_argument(
        "--window-ms-list",
        default="50,100",
        help="Comma-separated scheduling windows for route2, e.g. 10,50,100",
    )
    parser.add_argument(
        "--route2-grouping",
        choices=["exact", "similarity"],
        default="similarity",
        help="Grouping used by route2 scheduling window",
    )
    parser.add_argument("--arrival-gap-ms", type=int, default=5, help="Synthetic request arrival interval")
    parser.add_argument(
        "--cost-per-token-ms",
        type=float,
        default=0.03,
        help="Cost proxy for one prefill token in estimated saved compute",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/xy68/kvcache_lab/data",
        help="Output directory for CSV results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call model; use deterministic latency proxy for logic validation",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent in-flight requests per scenario")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N completed requests")
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def token_count(text: str) -> int:
    return len(tokenize(text))


def simhash(tokens: list[str]) -> int:
    if not tokens:
        return 0
    weights: dict[str, int] = {}
    for tok in tokens:
        weights[tok] = weights.get(tok, 0) + 1

    bits = [0] * 64
    for tok, wt in weights.items():
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(64):
            bits[i] += wt if ((h >> i) & 1) else -wt

    out = 0
    for i, v in enumerate(bits):
        if v >= 0:
            out |= 1 << i
    return out


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def load_items(prompt_file: Path, sample_size: int) -> list[PromptItem]:
    data = json.loads(prompt_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Prompt file must be a list: {prompt_file}")

    items: list[PromptItem] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        if "id" not in row or "prompt" not in row:
            continue
        items.append(
            PromptItem(
                req_id=int(row["id"]),
                prompt=str(row["prompt"]),
                prefix_type=str(row.get("prefix_type", "NA")),
            )
        )

    items.sort(key=lambda x: x.req_id)
    if sample_size > 0:
        items = items[:sample_size]
    if not items:
        raise ValueError("No valid prompts loaded")
    return items


def exact_group_key(prompt: str, prefix_chars: int) -> str:
    prefix = prompt[:prefix_chars]
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:16]


def similarity_group_keys(items: list[PromptItem], sim_prefix_tokens: int, threshold_bits: int) -> dict[int, str]:
    cluster_representative_hash: list[int] = []
    cluster_members: list[list[int]] = []
    out: dict[int, str] = {}

    for item in items:
        toks = tokenize(item.prompt)[:sim_prefix_tokens]
        h = simhash(toks)

        best_idx = -1
        best_dist = 10**9
        for idx, rep in enumerate(cluster_representative_hash):
            d = hamming_distance(h, rep)
            if d < best_dist:
                best_dist = d
                best_idx = idx

        if best_idx >= 0 and best_dist <= threshold_bits:
            cluster_members[best_idx].append(h)
            rep_hash = 0
            for bit in range(64):
                ones = sum((v >> bit) & 1 for v in cluster_members[best_idx])
                zeros = len(cluster_members[best_idx]) - ones
                if ones >= zeros:
                    rep_hash |= 1 << bit
            cluster_representative_hash[best_idx] = rep_hash
            out[item.req_id] = f"sim_{best_idx:03d}"
        else:
            cluster_members.append([h])
            cluster_representative_hash.append(h)
            out[item.req_id] = f"sim_{len(cluster_representative_hash)-1:03d}"

    return out


def build_group_keys(
    items: list[PromptItem],
    grouping: str,
    exact_prefix_chars: int,
    sim_prefix_tokens: int,
    sim_threshold_bits: int,
) -> dict[int, str]:
    if grouping == "fifo":
        return {item.req_id: f"fifo_{item.req_id}" for item in items}
    if grouping == "exact":
        return {item.req_id: exact_group_key(item.prompt, exact_prefix_chars) for item in items}
    if grouping == "similarity":
        return similarity_group_keys(items, sim_prefix_tokens, sim_threshold_bits)
    raise ValueError(f"Unsupported grouping: {grouping}")


def reorder_global_by_group(items: list[PromptItem], group_keys: dict[int, str]) -> list[PromptItem]:
    grouped: OrderedDict[str, list[PromptItem]] = OrderedDict()
    for item in items:
        g = group_keys[item.req_id]
        grouped.setdefault(g, []).append(item)

    out: list[PromptItem] = []
    for bucket in grouped.values():
        out.extend(bucket)
    return out


def reorder_bucket_by_group(bucket: list[PromptItem], group_keys: dict[int, str]) -> list[PromptItem]:
    grouped: OrderedDict[str, list[PromptItem]] = OrderedDict()
    for item in bucket:
        g = group_keys[item.req_id]
        grouped.setdefault(g, []).append(item)

    out: list[PromptItem] = []
    for bucket_items in grouped.values():
        out.extend(bucket_items)
    return out


def schedule_with_window(
    items: list[PromptItem],
    group_keys: dict[int, str],
    window_ms: int,
    arrival_gap_ms: int,
) -> tuple[list[PromptItem], dict[int, float]]:
    if window_ms <= 0:
        return list(items), {item.req_id: 0.0 for item in items}

    arrivals = {item.req_id: idx * arrival_gap_ms for idx, item in enumerate(items)}

    i = 0
    scheduled: list[PromptItem] = []
    wait_ms: dict[int, float] = {}

    while i < len(items):
        window_start = arrivals[items[i].req_id]
        cutoff = window_start + window_ms

        j = i
        bucket: list[PromptItem] = []
        while j < len(items) and arrivals[items[j].req_id] <= cutoff:
            bucket.append(items[j])
            j += 1

        reordered = reorder_bucket_by_group(bucket, group_keys)
        for item in reordered:
            delay = max(0.0, float(cutoff - arrivals[item.req_id]))
            wait_ms[item.req_id] = delay
        scheduled.extend(reordered)
        i = j

    return scheduled, wait_ms


def build_scenarios(window_ms_list: list[int], route2_grouping: str) -> list[Scenario]:
    scenarios = [
        Scenario(name="fifo_w0", route="route1", grouping="fifo", window_ms=0),
        Scenario(name="exact_w0", route="route1", grouping="exact", window_ms=0),
        Scenario(name="similarity_w0", route="route1", grouping="similarity", window_ms=0),
    ]

    for w in window_ms_list:
        if w > 0:
            scenarios.append(
                Scenario(
                    name=f"{route2_grouping}_window_{w}ms",
                    route="route2",
                    grouping=route2_grouping,
                    window_ms=w,
                )
            )
    return scenarios


def parse_window_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


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


def dry_run_once(prompt_tokens: int, reused_tokens: int, max_tokens: int) -> tuple[float, int]:
    base = 0.06 + 0.0015 * prompt_tokens
    gain = 0.00045 * reused_tokens
    latency = max(0.03, base - gain)
    out_tokens = max(8, min(max_tokens, prompt_tokens // 2))
    return latency, out_tokens


def summarize_latency(results: list[RequestResult]) -> dict[str, float]:
    vals = [r.total_latency_s for r in results]
    tps = [
        (r.output_tokens / r.total_latency_s)
        if r.total_latency_s > 0 and r.output_tokens > 0
        else 0.0
        for r in results
    ]
    return {
        "count": float(len(results)),
        "latency_mean_s": statistics.mean(vals) if vals else 0.0,
        "latency_p50_s": statistics.median(vals) if vals else 0.0,
        "latency_p90_s": statistics.quantiles(vals, n=10)[8] if len(vals) >= 10 else max(vals, default=0.0),
        "latency_min_s": min(vals, default=0.0),
        "latency_max_s": max(vals, default=0.0),
        "throughput_tokens_per_s_mean": statistics.mean(tps) if tps else 0.0,
    }


def write_request_csv(rows: list[RequestResult], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "route",
            "grouping",
            "window_ms",
            "order_index",
            "req_id",
            "prefix_type",
            "group_key",
            "schedule_wait_ms",
            "prompt_tokens",
            "prefix_tokens",
            "reused_tokens",
            "reuse_hit",
            "model_latency_s",
            "total_latency_s",
            "output_tokens",
            "throughput_tokens_per_s",
        ])
        for r in rows:
            tps = (r.output_tokens / r.total_latency_s) if r.total_latency_s > 0 and r.output_tokens > 0 else 0.0
            writer.writerow([
                r.scenario,
                r.route,
                r.grouping,
                r.window_ms,
                r.order_index,
                r.req_id,
                r.prefix_type,
                r.group_key,
                f"{r.schedule_wait_ms:.3f}",
                r.prompt_tokens,
                r.prefix_tokens,
                r.reused_tokens,
                r.reuse_hit,
                f"{r.model_latency_s:.6f}",
                f"{r.total_latency_s:.6f}",
                r.output_tokens,
                f"{tps:.6f}",
            ])


def write_summary_csv(summary_rows: list[dict], out_path: Path) -> None:
    fields = [
        "scenario",
        "route",
        "grouping",
        "window_ms",
        "count",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p90_s",
        "latency_min_s",
        "latency_max_s",
        "throughput_tokens_per_s_mean",
        "prefix_reuse_ratio",
        "shared_prefix_tokens",
        "total_prefix_tokens",
        "estimated_saved_compute_ms",
        "effective_latency_gain_per_prefix_token_s",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    window_ms_list = parse_window_list(args.window_ms_list)
    scenarios = build_scenarios(window_ms_list, args.route2_grouping)

    items = load_items(Path(args.prompt_file), args.sample_size)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key) if not args.dry_run else None

    if args.warmup > 0 and not args.dry_run:
        for _ in range(args.warmup):
            run_once(client, args.model, items[0].prompt, min(args.max_tokens, 64), args.temperature)

    all_results: list[RequestResult] = []
    summary_rows: list[dict] = []
    baseline_mean_latency = None

    for scenario in scenarios:
        group_keys = build_group_keys(
            items,
            grouping=scenario.grouping,
            exact_prefix_chars=args.exact_prefix_chars,
            sim_prefix_tokens=args.sim_prefix_tokens,
            sim_threshold_bits=args.sim_threshold_bits,
        )

        if scenario.route == "route1":
            if scenario.grouping == "fifo":
                ordered_items = list(items)
            else:
                ordered_items = reorder_global_by_group(items, group_keys)
            wait_by_req_id = {item.req_id: 0.0 for item in ordered_items}
        else:
            ordered_items, wait_by_req_id = schedule_with_window(
                items,
                group_keys=group_keys,
                window_ms=scenario.window_ms,
                arrival_gap_ms=args.arrival_gap_ms,
            )

        seen_groups: set[str] = set()
        scenario_results: list[RequestResult] = []
        shared_prefix_tokens = 0
        total_prefix_tokens = 0

        print(f"Running {scenario.name}: grouping={scenario.grouping}, window_ms={scenario.window_ms}")
        prepared: list[dict] = []
        for idx, item in enumerate(ordered_items):
            prompt_tokens = token_count(item.prompt)
            prefix_tokens = min(args.sim_prefix_tokens, prompt_tokens)
            group_key = group_keys[item.req_id]
            reuse_hit = 1 if group_key in seen_groups else 0
            reused_tokens = prefix_tokens if reuse_hit else 0
            schedule_wait_ms = wait_by_req_id.get(item.req_id, 0.0)

            prepared.append(
                {
                    "scenario": scenario.name,
                    "route": scenario.route,
                    "grouping": scenario.grouping,
                    "window_ms": scenario.window_ms,
                    "order_index": idx,
                    "req_id": item.req_id,
                    "prefix_type": item.prefix_type,
                    "group_key": group_key,
                    "schedule_wait_ms": schedule_wait_ms,
                    "prompt_tokens": prompt_tokens,
                    "prefix_tokens": prefix_tokens,
                    "reused_tokens": reused_tokens,
                    "reuse_hit": reuse_hit,
                    "prompt": item.prompt,
                }
            )

            total_prefix_tokens += prefix_tokens
            shared_prefix_tokens += reused_tokens
            seen_groups.add(group_key)

        completed = 0
        results_by_index: dict[int, tuple[float, int]] = {}

        if args.dry_run:
            for p in prepared:
                lat, out_tokens = dry_run_once(p["prompt_tokens"], p["reused_tokens"], args.max_tokens)
                results_by_index[p["order_index"]] = (lat, out_tokens)
                completed += 1
                if completed % max(1, args.progress_every) == 0 or completed == len(prepared):
                    print(f"  progress {completed}/{len(prepared)}", flush=True)
        else:
            workers = max(1, args.concurrency)
            print(f"  dispatching {len(prepared)} requests with concurrency={workers}", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_index = {
                    executor.submit(
                        run_once,
                        client,
                        args.model,
                        p["prompt"],
                        args.max_tokens,
                        args.temperature,
                    ): p["order_index"]
                    for p in prepared
                }

                for future in as_completed(future_to_index):
                    order_index = future_to_index[future]
                    try:
                        lat, out_tokens = future.result()
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            f"Scenario {scenario.name} failed at order_index={order_index}: {exc}"
                        ) from exc

                    results_by_index[order_index] = (lat, out_tokens)
                    completed += 1
                    if completed % max(1, args.progress_every) == 0 or completed == len(prepared):
                        print(f"  progress {completed}/{len(prepared)}", flush=True)

        for p in prepared:
            model_latency_s, out_tokens = results_by_index[p["order_index"]]
            total_latency_s = model_latency_s + (p["schedule_wait_ms"] / 1000.0)

            scenario_results.append(
                RequestResult(
                    scenario=p["scenario"],
                    route=p["route"],
                    grouping=p["grouping"],
                    window_ms=p["window_ms"],
                    order_index=p["order_index"],
                    req_id=p["req_id"],
                    prefix_type=p["prefix_type"],
                    group_key=p["group_key"],
                    schedule_wait_ms=p["schedule_wait_ms"],
                    prompt_tokens=p["prompt_tokens"],
                    prefix_tokens=p["prefix_tokens"],
                    reused_tokens=p["reused_tokens"],
                    reuse_hit=p["reuse_hit"],
                    model_latency_s=model_latency_s,
                    total_latency_s=total_latency_s,
                    output_tokens=out_tokens,
                )
            )

        all_results.extend(scenario_results)
        lat = summarize_latency(scenario_results)

        if scenario.name == "fifo_w0":
            baseline_mean_latency = lat["latency_mean_s"]

        reuse_ratio = (shared_prefix_tokens / total_prefix_tokens) if total_prefix_tokens > 0 else 0.0
        est_saved_compute_ms = shared_prefix_tokens * args.cost_per_token_ms
        avg_prefix_tokens = (total_prefix_tokens / len(scenario_results)) if scenario_results else 0.0

        if baseline_mean_latency is not None and avg_prefix_tokens > 0:
            gain_per_prefix_token = (baseline_mean_latency - lat["latency_mean_s"]) / avg_prefix_tokens
        else:
            gain_per_prefix_token = 0.0

        summary_rows.append(
            {
                "scenario": scenario.name,
                "route": scenario.route,
                "grouping": scenario.grouping,
                "window_ms": scenario.window_ms,
                "count": int(lat["count"]),
                "latency_mean_s": f"{lat['latency_mean_s']:.6f}",
                "latency_p50_s": f"{lat['latency_p50_s']:.6f}",
                "latency_p90_s": f"{lat['latency_p90_s']:.6f}",
                "latency_min_s": f"{lat['latency_min_s']:.6f}",
                "latency_max_s": f"{lat['latency_max_s']:.6f}",
                "throughput_tokens_per_s_mean": f"{lat['throughput_tokens_per_s_mean']:.6f}",
                "prefix_reuse_ratio": f"{reuse_ratio:.6f}",
                "shared_prefix_tokens": shared_prefix_tokens,
                "total_prefix_tokens": total_prefix_tokens,
                "estimated_saved_compute_ms": f"{est_saved_compute_ms:.6f}",
                "effective_latency_gain_per_prefix_token_s": f"{gain_per_prefix_token:.8f}",
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    request_csv = out_dir / f"prefix_similarity_requests_{ts}.csv"
    summary_csv = out_dir / f"prefix_similarity_summary_{ts}.csv"
    meta_json = out_dir / f"prefix_similarity_meta_{ts}.json"

    write_request_csv(all_results, request_csv)
    write_summary_csv(summary_rows, summary_csv)
    meta_json.write_text(
        json.dumps(
            {
                "config": vars(args),
                "input_count": len(items),
                "scenarios": [s.__dict__ for s in scenarios],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Done ===")
    print(f"Request CSV: {request_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Meta JSON: {meta_json}")


if __name__ == "__main__":
    main()
