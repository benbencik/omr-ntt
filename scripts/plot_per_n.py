#!/usr/bin/env python3
# Bar chart per N with all encoders side by side.

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from load_criterion import load_results


def plot(df, out: pathlib.Path) -> None:
    log2_ns = sorted(df["log2_N"].unique())
    n_groups = len(log2_ns)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 6), sharey=False)
    if n_groups == 1:
        axes = [axes]

    for ax, log2_n in zip(axes, log2_ns):
        sub = df[df["log2_N"] == log2_n].sort_values("median_ms")
        x = np.arange(len(sub))
        medians = sub["median_ms"].to_numpy()
        err_low = medians - sub["ci_low_ms"].to_numpy()
        err_high = sub["ci_high_ms"].to_numpy() - medians

        # ax.bar(x, medians, yerr=[err_low, err_high], capsize=4, color="steelblue", alpha=0.8)
        ax.bar(x, medians, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["encoder"].tolist(), rotation=40, ha="right", fontsize=9)
        ax.set_title(f"N = 2^{log2_n}")
        ax.set_ylabel("median time (ms)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("NTT encoder comparison — ntt_full", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    criterion_dir = sys.argv[1] if len(sys.argv) > 1 else "target/criterion"
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("per_n_bar.png")
    plot(load_results(criterion_dir), out)
