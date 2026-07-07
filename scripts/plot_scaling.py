#!/usr/bin/env python3
# Scaling plot: time vs log2(N) per encoder

import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from load_criterion import load_results


def plot(df, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for encoder, grp in df.groupby("encoder"):
        grp = grp.sort_values("log2_N")
        ax.plot(grp["log2_N"], grp["median_ms"], marker="o", label=encoder)
        ax.fill_between(grp["log2_N"], grp["ci_low_ms"], grp["ci_high_ms"], alpha=0.15)

    ax.set_yscale('log')
    ax.set_xlabel("log₂(N)")
    ax.set_ylabel("median time (ms)")
    ax.set_title("NTT encoder scaling — ntt_full")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    criterion_dir = sys.argv[1] if len(sys.argv) > 1 else "target/criterion"
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("scaling.png")
    plot(load_results(criterion_dir), out)
