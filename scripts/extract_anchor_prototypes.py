#!/usr/bin/env python3
"""
Extract PointPillars anchor prototypes (n_anchors x 7) from a decoded-head ONNX.

This utility searches initializers for arrays with size n_anchors * 7 (default 266),
selects the best candidate, and writes a flat float file that ROS can ingest via
anchor_prototypes_file.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple


def load_onnx_initializers(path: str):
    try:
        import onnx
        import onnx.numpy_helper as nph
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'onnx'. Install it with: pip install onnx"
        ) from exc

    model = onnx.load(path)
    for init in model.graph.initializer:
        arr = nph.to_array(init)
        yield init.name, arr


def pick_proto(initializers, expected_size: int, expected_rows: Optional[int]) -> Tuple[str, List[float]]:
    candidates = []
    for name, arr in initializers:
        if arr.size != expected_size:
            continue
        score = 0
        if arr.ndim == 2 and arr.shape[1] == 7:
            score += 100
        if expected_rows is not None and arr.ndim == 2 and arr.shape[0] == expected_rows and arr.shape[1] == 7:
            score += 100
        if "anchor" in name.lower() or "proto" in name.lower():
            score += 10
        candidates.append((score, name, arr))

    if not candidates:
        raise ValueError(
            f"No initializer found with size={expected_size}. "
            "Provide the decoded-head ONNX or pass a different expected size."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, name, arr = candidates[0]

    if arr.ndim == 1:
        if expected_rows is None:
            expected_rows = arr.size // 7
        arr = arr.reshape(expected_rows, 7)
    elif arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(
            f"Selected initializer '{name}' has shape {arr.shape}, expected (*,7)."
        )

    flat = arr.astype("float32").reshape(-1).tolist()
    return name, flat


def write_flat_txt(path: str, values: List[float], row_width: int = 7):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# anchor_prototypes flattened values\n")
        for i in range(0, len(values), row_width):
            row = values[i : i + row_width]
            f.write(" ".join(f"{v:.8f}" for v in row) + "\n")


def print_yaml(values: List[float]):
    joined = ", ".join(f"{v:.8f}" for v in values)
    print(f"anchor_prototypes: [{joined}]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded_onnx", required=True, help="Path to decoded-head ONNX")
    p.add_argument(
        "--expected_size",
        type=int,
        default=266,
        help="Expected flattened size for prototype initializer (default 266 for 38x7)",
    )
    p.add_argument(
        "--expected_rows",
        type=int,
        default=38,
        help="Expected row count, used when reshaping 1D tensors (default 38)",
    )
    p.add_argument(
        "--output",
        default="",
        help="Output anchor txt path. If omitted, prints YAML list only.",
    )
    p.add_argument(
        "--print_yaml",
        action="store_true",
        help="Print ROS YAML inline list for anchor_prototypes",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.decoded_onnx):
        print(f"ERROR: ONNX not found: {args.decoded_onnx}", file=sys.stderr)
        return 2

    try:
        name, flat = pick_proto(
            load_onnx_initializers(args.decoded_onnx),
            expected_size=args.expected_size,
            expected_rows=args.expected_rows,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    print(f"Selected initializer: {name}")
    print(f"Values: {len(flat)}")

    if args.output:
        write_flat_txt(args.output, flat, row_width=7)
        print(f"Wrote: {args.output}")

    if args.print_yaml or not args.output:
        print_yaml(flat)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
