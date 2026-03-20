#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

    numeric_columns = [
        "utilization.gpu [%]",
        "utilization.memory [%]",
        "memory.used [MiB]",
        "memory.total [MiB]",
        "temperature.gpu",
    ]
    for column in numeric_columns:
        df[column] = _parse_numeric(df[column])

    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def plot_gpu_log(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    elapsed_minutes = df.index * 5 / 60  # 每条记录间隔5秒，转换为分钟

    axes[0].plot(elapsed_minutes, df["utilization.gpu [%]"], label="GPU Utilization", linewidth=2)
    axes[0].plot(elapsed_minutes, df["utilization.memory [%]"], label="Memory Utilization", linewidth=2)
    axes[0].set_ylabel("Utilization (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(elapsed_minutes, df["memory.used [MiB]"], color="tab:orange", linewidth=2, label="Memory Used")
    if df["memory.total [MiB]"].notna().any():
        total_memory = df["memory.total [MiB]"].dropna().iloc[0]
        axes[1].axhline(total_memory, color="tab:red", linestyle="--", alpha=0.7, label="Memory Total")
    axes[1].set_ylabel("Memory (MiB)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(elapsed_minutes, df["temperature.gpu"], color="tab:green", linewidth=2)
    axes[2].set_ylabel("Temperature (C)")
    axes[2].set_xlabel("Training Duration (min)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("GPU Log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GPU usage logs exported to CSV.")
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to the input CSV file.",
    )
    args = parser.parse_args()

    input_path = args.path.expanduser().resolve()
    output_path = input_path.with_suffix(".png")

    df = load_gpu_log(input_path)
    plot_gpu_log(df, output_path)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
