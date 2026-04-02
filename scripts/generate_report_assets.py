#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_latest_file(base: Path, pattern: str) -> Path:
    files = sorted(base.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matches: {pattern}")
    return files[-1]


def build_cache_reports(base: Path, plots: Path) -> None:
    cache_req_file = load_latest_file(base, "prefix_cache_requests_*.csv")
    cache_req = pd.read_csv(cache_req_file)

    cache_req["latency_s"] = pd.to_numeric(cache_req["latency_s"])
    cache_req["throughput_tokens_per_s"] = pd.to_numeric(cache_req["throughput_tokens_per_s"])

    cache_group = (
        cache_req.groupby("mode")
        .agg(
            count=("mode", "size"),
            latency_mean_s=("latency_s", "mean"),
            latency_p50_s=("latency_s", "median"),
            latency_p90_s=("latency_s", lambda x: np.percentile(x, 90)),
            latency_min_s=("latency_s", "min"),
            latency_max_s=("latency_s", "max"),
            throughput_tokens_per_s_mean=("throughput_tokens_per_s", "mean"),
            output_tokens_mean=("output_tokens", "mean"),
        )
        .reset_index()
    )

    cache_group.to_csv(base / "report_cache_group_stats.csv", index=False)

    plt.figure(figsize=(8, 4.8), dpi=140)
    for mode, color in [("shared_prefix", "#1f77b4"), ("unique_prefix", "#ff7f0e")]:
        vals = cache_req.loc[cache_req["mode"] == mode, "latency_s"].values
        x = np.random.normal(0, 0.03, size=len(vals)) + (0 if mode == "shared_prefix" else 1)
        plt.scatter(x, vals, alpha=0.45, s=22, color=color, label=mode)

    means = [
        cache_req.loc[cache_req["mode"] == "shared_prefix", "latency_s"].mean(),
        cache_req.loc[cache_req["mode"] == "unique_prefix", "latency_s"].mean(),
    ]
    plt.plot([0, 1], means, color="#222222", marker="o", linewidth=1.8, label="mean")
    plt.xticks([0, 1], ["shared_prefix", "unique_prefix"])
    plt.ylabel("Latency (s)")
    plt.title("Prefix Cache: Shared vs Unique Prefix Latency")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(plots / "report_cache_shared_vs_unique_latency.png")
    plt.close()


def build_similarity_reports(base: Path, plots: Path) -> None:
    sim_sum_files = sorted(base.glob("prefix_similarity_summary_*.csv"))
    sim_req_files = sorted(base.glob("prefix_similarity_requests_*.csv"))
    if not sim_sum_files or not sim_req_files:
        raise FileNotFoundError("Missing similarity summary/request csv files")

    sim_summaries = []
    for f in sim_sum_files:
        ts = f.stem.split("_")[-1]
        df = pd.read_csv(f)
        df["run_ts"] = ts
        sim_summaries.append(df)

    sim_sum = pd.concat(sim_summaries, ignore_index=True)

    num_cols = [
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
    for c in num_cols:
        sim_sum[c] = pd.to_numeric(sim_sum[c])

    scenario_order = ["fifo_w0", "exact_w0", "similarity_w0", "similarity_window_50ms", "similarity_window_100ms"]
    sim_sum["scenario"] = pd.Categorical(sim_sum["scenario"], categories=scenario_order, ordered=True)
    sim_sum = sim_sum.sort_values(["scenario", "run_ts"])

    agg = (
        sim_sum.groupby("scenario", observed=False)
        .agg(
            runs=("latency_mean_s", "size"),
            latency_mean_avg_s=("latency_mean_s", "mean"),
            latency_mean_std_s=("latency_mean_s", "std"),
            latency_p90_avg_s=("latency_p90_s", "mean"),
            throughput_avg=("throughput_tokens_per_s_mean", "mean"),
            reuse_ratio_avg=("prefix_reuse_ratio", "mean"),
        )
        .reset_index()
    )

    fifo_by_run = sim_sum[sim_sum["scenario"] == "fifo_w0"][["run_ts", "latency_mean_s"]].rename(
        columns={"latency_mean_s": "fifo_latency_mean_s"}
    )
    rel = sim_sum.merge(fifo_by_run, on="run_ts", how="left")
    rel["latency_improve_vs_fifo_pct"] = (
        (rel["fifo_latency_mean_s"] - rel["latency_mean_s"]) / rel["fifo_latency_mean_s"] * 100
    )

    rel_agg = (
        rel.groupby("scenario", observed=False)
        .agg(
            improve_vs_fifo_pct_avg=("latency_improve_vs_fifo_pct", "mean"),
            improve_vs_fifo_pct_std=("latency_improve_vs_fifo_pct", "std"),
        )
        .reset_index()
    )

    final_agg = agg.merge(rel_agg, on="scenario", how="left").sort_values("scenario")
    final_agg.to_csv(base / "report_similarity_scenario_aggregate.csv", index=False)

    plot_df = final_agg.set_index("scenario").loc[scenario_order].reset_index()

    plt.figure(figsize=(9.4, 5.2), dpi=140)
    plt.bar(
        plot_df["scenario"],
        plot_df["latency_mean_avg_s"],
        yerr=plot_df["latency_mean_std_s"].fillna(0),
        color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
        capsize=4,
    )
    plt.ylabel("Mean Total Latency (s)")
    plt.title("Prefix Similarity Benchmark: Scenario Latency Across Runs")
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots / "report_similarity_scenario_mean_latency.png")
    plt.close()

    pivot = sim_sum.pivot(index="run_ts", columns="scenario", values="latency_mean_s")
    pivot = pivot[scenario_order]

    plt.figure(figsize=(9.4, 5.2), dpi=140)
    for run_ts in pivot.index:
        plt.plot(scenario_order, pivot.loc[run_ts].values, marker="o", linewidth=1.6, alpha=0.85, label=str(run_ts))
    plt.ylabel("Mean Total Latency (s)")
    plt.title("Per-Run Scenario Variation")
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(title="run", frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(plots / "report_similarity_run_variation.png")
    plt.close()

    plt.figure(figsize=(9.2, 5.0), dpi=140)
    plt.bar(
        plot_df["scenario"],
        plot_df["improve_vs_fifo_pct_avg"],
        yerr=plot_df["improve_vs_fifo_pct_std"].fillna(0),
        color=["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
        capsize=4,
    )
    plt.axhline(0, color="#333333", linewidth=1)
    plt.ylabel("Latency Improvement vs FIFO (%)")
    plt.title("Scenario Improvement Relative to FIFO")
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots / "report_similarity_improve_vs_fifo.png")
    plt.close()

    all_req = []
    for f in sim_req_files:
        ts = f.stem.split("_")[-1]
        df = pd.read_csv(f)
        df["run_ts"] = ts
        all_req.append(df)
    req = pd.concat(all_req, ignore_index=True)

    for c in ["schedule_wait_ms", "model_latency_s", "total_latency_s", "reused_tokens"]:
        req[c] = pd.to_numeric(req[c])

    route2 = req[req["route"] == "route2"].copy()
    wait_stats = (
        route2.groupby("window_ms")
        .agg(
            count=("window_ms", "size"),
            schedule_wait_ms_mean=("schedule_wait_ms", "mean"),
            model_latency_s_mean=("model_latency_s", "mean"),
            total_latency_s_mean=("total_latency_s", "mean"),
        )
        .reset_index()
        .sort_values("window_ms")
    )
    wait_stats.to_csv(base / "report_window_wait_stats.csv", index=False)

    plt.figure(figsize=(8.2, 5.0), dpi=140)
    for w, color in [(50, "#ff7f0e"), (100, "#d62728")]:
        sub = route2[route2["window_ms"] == w]
        plt.scatter(sub["schedule_wait_ms"], sub["total_latency_s"], s=14, alpha=0.35, label=f"window {w}ms", color=color)
    plt.xlabel("Schedule Wait (ms)")
    plt.ylabel("Total Latency (s)")
    plt.title("Route2 Windowing: Wait vs Total Latency")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(plots / "report_window_wait_vs_total_latency.png")
    plt.close()


def build_prompt_reports(root: Path, base: Path, plots: Path) -> None:
    prompt_file = root / "vllm_shared_prefix_10types_100.json"
    prompt_data = json.loads(prompt_file.read_text(encoding="utf-8"))
    prompt_df = pd.DataFrame(prompt_data)

    prompt_stats = (
        prompt_df.groupby("prefix_type")
        .agg(count=("id", "size"))
        .reset_index()
        .sort_values("prefix_type")
    )
    prompt_stats.to_csv(base / "report_prompt_prefix_type_counts.csv", index=False)

    plt.figure(figsize=(7.8, 4.8), dpi=140)
    plt.bar(prompt_stats["prefix_type"], prompt_stats["count"], color="#1f77b4")
    plt.xlabel("Prefix Type")
    plt.ylabel("Count")
    plt.title("Prompt Prefix Type Distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots / "report_prompt_prefix_type_counts.png")
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    base = root / "data"
    plots = root / "plots"
    plots.mkdir(exist_ok=True)

    build_cache_reports(base, plots)
    build_similarity_reports(base, plots)
    build_prompt_reports(root, base, plots)

    print("Generated reports in:", base)
    print("Generated plots in:", plots)


if __name__ == "__main__":
    main()
