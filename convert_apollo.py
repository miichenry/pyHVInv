#!/usr/bin/env python3
"""
Convert apollo/ DC files from period (s) / km/s to frequency (Hz) / m/s.
HV files are copied unchanged. Output goes to apollo_converted/.

Usage
-----
    python convert_apollo.py
    python convert_apollo.py --input apollo --output apollo_converted
"""

import argparse
import shutil
import numpy as np
from pathlib import Path


def convert_dc(src: Path, dst: Path):
    data = np.loadtxt(src, comments="#")
    data[:, 0] = 1.0 / data[:, 0]        # period (s) → frequency (Hz)
    data[:, 1] = data[:, 1] * 1000.0     # km/s → m/s
    if data.shape[1] >= 3:
        data[:, 2] = data[:, 2] * 1000.0 # std km/s → m/s

    with open(dst, "w") as f:
        f.write("# frequency(Hz) Vg(m/s)\n")
        np.savetxt(f, data, fmt="%.10e")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="apollo",           help="Source directory")
    parser.add_argument("--output", default="apollo_converted", help="Destination directory")
    args = parser.parse_args()

    src_dir = Path(args.input)
    dst_dir = Path(args.output)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(src_dir.iterdir()):
        if f.name.startswith("disp_") and f.suffix == ".dat":
            convert_dc(f, dst_dir / f.name)
            print(f"Converted : {f.name}")
        elif f.name.startswith("hvsr_") and f.suffix == ".dat":
            shutil.copy(f, dst_dir / f.name)
            print(f"Copied    : {f.name}")


if __name__ == "__main__":
    main()
