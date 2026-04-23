#!/usr/bin/env python3
"""
Batch runner: invoke run_inversion_v3.py for every matched HV/DC pair in a directory.

Usage
-----
    python run_batch.py                        # process apollo/ with default settings
    python run_batch.py --data-dir apollo      # explicit data directory
    python run_batch.py --data-dir apollo --output-dir observed  # custom output root

Output
------
Each site writes its results to <output-dir>/<coord>/, plus:
    <output-dir>/figures/<coord>_inversion.png   ← all figures in one place
    <output-dir>/summary.csv                     ← misfit + model params for all sites
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def find_pairs(data_dir: Path):
    """Return list of (coord, hv_path, dc_path) for matched files."""
    pairs = []
    for hv_file in sorted(data_dir.glob("hvsr_*.dat")):
        coord = hv_file.stem[len("hvsr_"):]          # e.g. 38.3797_14.97612
        dc_file = data_dir / f"disp_{coord}_td.dat"
        if dc_file.exists():
            pairs.append((coord, hv_file, dc_file))
        else:
            print(f"[WARN] No matching DC file for {hv_file.name}, skipping.")
    return pairs


def build_csv_row(coord: str, summary: dict) -> dict:
    row = {"coord": coord, "misfit": summary["misfit"]}
    for k, layer in enumerate(summary["layers"]):
        label = "hs" if k == summary["n_layers"] - 1 else str(k + 1)
        for param in ("h", "vp", "vs", "rho", "nu"):
            row[f"{param}_{label}"] = layer.get(param)
    return row


def main():
    parser = argparse.ArgumentParser(description="Batch HV-DC inversion over a directory")
    parser.add_argument("--data-dir", default="apollo", help="Directory with HV/DC files")
    parser.add_argument("--output-dir", default="observed", help="Root output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_root = Path(args.output_dir)
    figures_dir = output_root / "figures"
    script = Path(__file__).parent / "run_inversion_v3.py"

    pairs = find_pairs(data_dir)
    if not pairs:
        print(f"No matched HV/DC pairs found in {data_dir}")
        sys.exit(1)

    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(pairs)} site(s) in '{data_dir}'. Output root: '{output_root}'\n")

    csv_rows = []
    failed = []

    for i, (coord, hv_file, dc_file) in enumerate(pairs, 1):
        out_dir = output_root / coord
        print(f"[{i}/{len(pairs)}] {coord}")
        print(f"  HV : {hv_file}")
        print(f"  DC : {dc_file}")
        print(f"  Out: {out_dir}")

        result = subprocess.run(
            [sys.executable, str(script),
             "--hv", str(hv_file),
             "--dc", str(dc_file),
             "--output-dir", str(out_dir),
             "--no-show"],
        )

        if result.returncode != 0:
            print(f"  [FAILED] returncode={result.returncode}\n")
            failed.append(coord)
            continue

        # Collect summary for CSV
        summary_file = out_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            csv_rows.append(build_csv_row(coord, summary))

        # Copy figure to top-level figures/
        fig_src = out_dir / "inversion_results.png"
        if fig_src.exists():
            shutil.copy(fig_src, figures_dir / f"{coord}_inversion.png")

        print(f"  [OK]\n")

    # Write summary CSV
    if csv_rows:
        csv_path = output_root / "summary.csv"
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Summary written to: {csv_path}")

    if failed:
        print(f"\n[WARN] {len(failed)} site(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
