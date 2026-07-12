# Load criterion benchmark results into a DataFrame and save to CSV.

import argparse
import json
import pathlib
import re

import pandas as pd


def load_results(criterion_dir: str | pathlib.Path = "target/criterion") -> pd.DataFrame:
    criterion_dir = pathlib.Path(criterion_dir)
    rows = []
    for est_path in criterion_dir.rglob("*/base/estimates.json"):
        # path: <criterion_dir>/<group>/<encoder>/base/estimates.json
        encoder = est_path.parts[-3]
        group = est_path.parts[-4]

        m = re.search(r"N=2_(\d+)", group)
        if not m:
            continue
        log2_n = int(m.group(1))

        data = json.loads(est_path.read_text())
        median_ns = data["median"]["point_estimate"]
        ci_low_ns = data["median"]["confidence_interval"]["lower_bound"]
        ci_high_ns = data["median"]["confidence_interval"]["upper_bound"]

        rows.append({
            "log2_N": log2_n,
            "N": 2 ** log2_n,
            "encoder": encoder,
            "median_ms": median_ns / 1e6,
            "ci_low_ms": ci_low_ns / 1e6,
            "ci_high_ms": ci_high_ns / 1e6,
        })

    if not rows:
        raise RuntimeError(f"No estimates.json found under {criterion_dir}")

    return pd.DataFrame(rows).sort_values(["log2_N", "encoder"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Load criterion results and save to CSV.")
    p.add_argument("criterion_dir", nargs="?", default="target/criterion",
                   help="Criterion output directory (default: target/criterion)")
    p.add_argument("--out", type=pathlib.Path, default="criterion_results.csv",
                   help="Output CSV path (default: criterion_results.csv)")
    args = p.parse_args()

    df = load_results(args.criterion_dir)
    df.to_csv(args.out, index=False, float_format="%.2f")
    print(f"Saved {len(df)} rows: {args.out}")


if __name__ == "__main__":
    main()
