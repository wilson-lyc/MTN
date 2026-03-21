#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    "utilization.gpu [%]",
    "utilization.memory [%]",
    "memory.used [MiB]",
    "memory.total [MiB]",
    "temperature.gpu",
]


def _parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("MiB", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_gpu_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")

    for column in NUMERIC_COLUMNS:
        df[column] = _parse_numeric(df[column])

    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def compute_global_mean(df: pd.DataFrame) -> pd.Series:
    return df[NUMERIC_COLUMNS].mean(numeric_only=True)


def compute_rolling_mean(
    df: pd.DataFrame,
    window_seconds: int,
    min_periods: int,
) -> pd.DataFrame:
    rolling = (
        df.set_index("timestamp")[NUMERIC_COLUMNS]
        .rolling(window=f"{window_seconds}s", min_periods=min_periods)
        .mean()
        .reset_index()
    )
    rolling = rolling.rename(
        columns={c: f"{c} (rolling_{window_seconds}s_mean)" for c in NUMERIC_COLUMNS}
    )
    return rolling


def compute_per_minute_mean(df: pd.DataFrame) -> pd.DataFrame:
    per_minute = (
        df.set_index("timestamp")[NUMERIC_COLUMNS]
        .resample("1min")
        .mean()
        .dropna(how="all")
        .reset_index()
    )
    return per_minute


def plot_means(
    df: pd.DataFrame,
    rolling_mean: pd.DataFrame,
    per_minute_mean: pd.DataFrame,
    global_mean: pd.Series,
    window_seconds: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    elapsed_minutes = df.index * 5 / 60
    start_ts = df["timestamp"].iloc[0]
    per_minute_elapsed = (per_minute_mean["timestamp"] - start_ts).dt.total_seconds() / 60.0

    axes[0].plot(elapsed_minutes, df["utilization.gpu [%]"], alpha=0.25, label="GPU Utilization Raw")
    axes[0].plot(
        elapsed_minutes,
        rolling_mean[f"utilization.gpu [%] (rolling_{window_seconds}s_mean)"],
        linewidth=2.2,
        label=f"GPU Utilization Rolling({window_seconds}s)",
    )
    axes[0].plot(
        per_minute_elapsed,
        per_minute_mean["utilization.gpu [%]"],
        linestyle="--",
        linewidth=1.8,
        label="GPU Utilization Per-Minute",
    )
    axes[0].axhline(
        global_mean["utilization.gpu [%]"],
        color="tab:red",
        linestyle=":",
        label="GPU Utilization Global Mean",
    )
    axes[0].set_ylabel("GPU Utilization (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(elapsed_minutes, df["memory.used [MiB]"], alpha=0.25, label="Memory Used Raw")
    axes[1].plot(
        elapsed_minutes,
        rolling_mean[f"memory.used [MiB] (rolling_{window_seconds}s_mean)"],
        linewidth=2.2,
        label=f"Memory Used Rolling({window_seconds}s)",
    )
    axes[1].plot(
        per_minute_elapsed,
        per_minute_mean["memory.used [MiB]"],
        linestyle="--",
        linewidth=1.8,
        label="Memory Used Per-Minute",
    )
    axes[1].axhline(
        global_mean["memory.used [MiB]"],
        color="tab:red",
        linestyle=":",
        label="Memory Used Global Mean",
    )
    axes[1].set_ylabel("Memory (MiB)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(elapsed_minutes, df["temperature.gpu"], alpha=0.25, label="Temperature Raw")
    axes[2].plot(
        elapsed_minutes,
        rolling_mean[f"temperature.gpu (rolling_{window_seconds}s_mean)"],
        linewidth=2.2,
        label=f"Temperature Rolling({window_seconds}s)",
    )
    axes[2].plot(
        per_minute_elapsed,
        per_minute_mean["temperature.gpu"],
        linestyle="--",
        linewidth=1.8,
        label="Temperature Per-Minute",
    )
    axes[2].axhline(
        global_mean["temperature.gpu"],
        color="tab:red",
        linestyle=":",
        label="Temperature Global Mean",
    )
    axes[2].set_ylabel("Temperature (C)")
    axes[2].set_xlabel("Training Duration (min)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("GPU Resource Usage Record")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GPU log means from CSV.")
    parser.add_argument("--path", type=Path, required=True, help="Path to input CSV.")
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=60,
        help="Rolling window size in seconds. Default: 60.",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=1,
        help="Minimum points required for rolling mean. Default: 1.",
    )
    args = parser.parse_args()

    input_path = args.path.expanduser().resolve()
    df = load_gpu_log(input_path)

    global_mean = compute_global_mean(df)
    rolling_mean = compute_rolling_mean(
        df,
        window_seconds=args.window_seconds,
        min_periods=args.min_periods,
    )
    per_minute_mean = compute_per_minute_mean(df)

    global_path = input_path.with_name(f"{input_path.stem}_global_mean.csv")
    plot_path = input_path.with_name(f"{input_path.stem}_resource_usage_record.png")

    global_mean.reset_index().rename(columns={"index": "metric", 0: "mean"}).to_csv(global_path, index=False)
    plot_means(
        df,
        rolling_mean,
        per_minute_mean,
        global_mean,
        window_seconds=args.window_seconds,
        output_path=plot_path,
    )

    print("=== Global Mean ===")
    for k, v in global_mean.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Saved Files ===")
    print(f"Global mean: {global_path}")
    print(f"Mean plot: {plot_path}")


if __name__ == "__main__":
    main()
