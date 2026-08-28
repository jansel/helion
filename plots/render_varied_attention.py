from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics

EXPECTED_IMPLEMENTATIONS = {
    "fa4",
    "flexattention-cute",
    "helion-cute",
    "sdpa",
}
CUTE_IMPLEMENTATIONS = EXPECTED_IMPLEMENTATIONS - {"sdpa"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the varied-shape BF16 attention comparison."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--evidence",
        type=Path,
        help="validate CSV medians against the tracked raw timing evidence",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    shape_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        shape_rows[int(row["shape_order"])].append(row)

    if sorted(shape_rows) != list(range(1, 9)):
        raise ValueError("expected shape_order values 1 through 8")
    for shape_order, group in shape_rows.items():
        if len(group) != len(EXPECTED_IMPLEMENTATIONS):
            raise ValueError(f"shape {shape_order} must have exactly four rows")
        implementations = {row["implementation"] for row in group}
        if implementations != EXPECTED_IMPLEMENTATIONS:
            raise ValueError(
                f"shape {shape_order} has implementations {implementations}"
            )
        shape_fields = {
            (
                row["shape_group"],
                row["variant"],
                row["z"],
                row["h"],
                row["seq_len"],
                row["head_dim"],
                row["dtype"],
                row["epilogue"],
            )
            for row in group
        }
        if len(shape_fields) != 1:
            raise ValueError(f"shape {shape_order} has inconsistent metadata")

    if {row["dtype"] for row in rows} != {"bfloat16"}:
        raise ValueError("the varied-shape report must contain only BF16 rows")
    if {row["gpu"] for row in rows} != {"NVIDIA B200"}:
        raise ValueError("the varied-shape report must contain only B200 rows")
    if {row["power_cap_w"] for row in rows} != {"750"}:
        raise ValueError("the varied-shape report must contain only 750 W rows")

    versions: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = float(row["tflops"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid TFLOP/s value: {row['tflops']}")
        versions[row["implementation"]].add(row["version"])
        labels[row["implementation"]].add(row["implementation_label"])
    for implementation in EXPECTED_IMPLEMENTATIONS:
        if len(versions[implementation]) != 1:
            raise ValueError(f"inconsistent {implementation} versions")
        if len(labels[implementation]) != 1:
            raise ValueError(f"inconsistent {implementation} labels")

    cute_versions: set[str] = set()
    for implementation in CUTE_IMPLEMENTATIONS:
        version = next(iter(versions[implementation]))
        matches = re.findall(r"(?:^|; )CuTe ([^;\s]+)", version)
        if len(matches) != 1:
            raise ValueError(f"missing CuTe version for {implementation}")
        cute_versions.add(matches[0])
    if len(cute_versions) != 1:
        raise ValueError(
            f"CuTe-backed implementations mix CuTe versions: {cute_versions}"
        )


def _validate_evidence(rows: list[dict[str, str]], path: Path) -> None:
    payload = json.loads(path.read_text())
    evidence = {
        (int(item["shape_order"]), item["implementation"]): item
        for item in payload["measurements"]
    }
    expected_keys = {(int(row["shape_order"]), row["implementation"]) for row in rows}
    if evidence.keys() != expected_keys:
        raise ValueError("evidence measurements do not match the CSV rows")

    for row in rows:
        key = (int(row["shape_order"]), row["implementation"])
        item = evidence[key]
        median_ms = statistics.median(float(value) for value in item["runs_ms"])
        computed_tflops = float(item["flop_count"]) / median_ms / 1e9
        if not math.isclose(
            computed_tflops,
            float(row["tflops"]),
            rel_tol=1e-15,
            abs_tol=1e-12,
        ):
            raise ValueError(f"evidence throughput does not match CSV row {key}")
        if len(item["runs_ms"]) != int(row["sample_count"]):
            raise ValueError(f"evidence sample count does not match CSV row {key}")
        if item["source_artifact"] != row["evidence_file"]:
            raise ValueError(f"evidence source does not match CSV row {key}")


def _shape_label(row: dict[str, str]) -> str:
    seq_len = int(row["seq_len"])
    seq_label = f"{seq_len // 1024}K" if seq_len % 1024 == 0 else f"{seq_len:,}"
    variant = row["variant"]
    if row["epilogue"]:
        variant += f" + {row['epilogue']}"
    return f"{variant}\n{row['z']}x{row['h']}\n{seq_label}x{row['head_dim']}"


def _caption_lines(rows: list[dict[str, str]]) -> list[str]:
    nonpassing = [row for row in rows if row["correctness"] != "PASS"]
    if not nonpassing:
        return []
    labels = sorted(
        {
            f"{row['implementation_label']} on shape {row['shape_order']} "
            f"({row['correctness']})"
            for row in nonpassing
        }
    )
    return ["Hatching marks non-PASS correctness: " + "; ".join(labels) + "."]


def _render(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows_by_shape: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        rows_by_shape[int(row["shape_order"])][row["implementation"]] = row

    shape_orders = sorted(rows_by_shape)
    values_by_impl = {
        implementation: [
            float(rows_by_shape[shape_order][implementation]["tflops"])
            for shape_order in shape_orders
        ]
        for implementation in EXPECTED_IMPLEMENTATIONS
    }
    plot_impls = sorted(
        EXPECTED_IMPLEMENTATIONS,
        key=lambda implementation: statistics.fmean(values_by_impl[implementation]),
    )

    first_rows = [
        next(iter(rows_by_shape[shape_order].values())) for shape_order in shape_orders
    ]
    labels = [_shape_label(row) for row in first_rows]
    x = np.arange(len(shape_orders))
    width = 0.82 / len(plot_impls)
    fig, ax = plt.subplots(figsize=(24.0, 9.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, implementation in enumerate(plot_impls):
        offsets = x + (index - (len(plot_impls) - 1) / 2) * width
        sample = rows_by_shape[shape_orders[0]][implementation]
        legend_label = f"{sample['implementation_label']}\n{sample['version']}"
        bars = ax.bar(
            offsets,
            values_by_impl[implementation],
            width,
            label=legend_label,
            color=colors[index % len(colors)],
        )
        for bar, shape_order in zip(bars, shape_orders, strict=True):
            result = rows_by_shape[shape_order][implementation]
            if result["correctness"] != "PASS":
                bar.set_hatch("///")
                bar.set_edgecolor("#303030")
                bar.set_linewidth(0.8)

    for separator in (1.5, 3.5, 5.5):
        ax.axvline(separator, color="#808080", linewidth=0.8, alpha=0.28)

    group_labels = (
        (0.5, "head dim 128"),
        (2.5, "batch 1"),
        (4.5, "batch 8"),
        (6.5, "ReLU epilogue"),
    )
    for center, label in group_labels:
        ax.text(
            center,
            -0.22,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
            fontweight="semibold",
        )

    ax.set_ylabel("Throughput (TFLOP/s)", fontsize=14)
    ax.set_title(
        "Attention forward throughput across varied BF16 shapes | "
        "NVIDIA B200 | 750 W power cap",
        fontsize=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, linespacing=1.15)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(ncols=1, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    captions = _caption_lines(rows)
    for index, caption in enumerate(captions):
        fig.text(
            0.01,
            0.022 + 0.023 * (len(captions) - index - 1),
            caption,
            ha="left",
            va="bottom",
            fontsize=10,
        )
    bottom = 0.075 if captions else 0.04
    fig.tight_layout(rect=(0, bottom, 0.78, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    rows = _read_rows(args.csv_path)
    _validate_rows(rows)
    if args.evidence is not None:
        _validate_evidence(rows, args.evidence)
    _render(rows, args.output_path)


if __name__ == "__main__":
    main()
