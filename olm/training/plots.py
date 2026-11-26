"""Dependency-free SVG learning-curve rendering for long CPU runs."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _thin(values: list[float], limit: int = 1200) -> list[float]:
    if len(values) <= limit:
        return values
    stride = len(values) / limit
    return [values[min(len(values) - 1, int(i * stride))] for i in range(limit)]


def _scale(values: list[float], top: float, height: float) -> list[float]:
    if not values:
        return []
    return [height - (value / top) * height for value in values]


def _polyline(values: list[float], width: float, height: float, top: float, color: str) -> str:
    values = _thin(values)
    if len(values) < 2:
        return ""
    ys = _scale(values, top, height)
    points = " ".join(f"{i * width / (len(values) - 1):.1f},{y:.1f}" for i, y in enumerate(ys))
    return f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}"/>'


def _panel(x: float, y: float, title: str, series: list[tuple[str, list[float], str]], width: float = 480, height: float = 250) -> str:
    parts = [f'<text x="{x}" y="{y - 10}" font-family="sans-serif" font-size="14" font-weight="600">{title}</text>']
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="none" stroke="#999"/>')
    all_values = [v for _, values, _ in series for v in values if math.isfinite(v)]
    if not all_values:
        parts.append(f'<text x="{x + 12}" y="{y + 24}" font-family="sans-serif" font-size="12">no data yet</text>')
        return "".join(parts)
    top = max(all_values)
    top = top if top > 0 else 1.0
    for index, (label, values, color) in enumerate(series):
        parts.append(_polyline(values, width, height, top, color))
        if values:
            parts.append(f'<text x="{x + 6}" y="{y + 14 + 14 * index}" font-family="sans-serif" font-size="11" fill="{color}">{label}</text>')
    parts.append(f'<text x="{x}" y="{y + height + 18}" font-family="sans-serif" font-size="11">0</text>')
    parts.append(f'<text x="{x + width - 30}" y="{y + height + 18}" font-family="sans-serif" font-size="11">step</text>')
    parts.append(f'<text x="{x}" y="{y + 12}" font-family="sans-serif" font-size="10" fill="#666">{top:.3g}</text>')
    return "".join(parts)


def render_learning_curves(batch_path: Path, epoch_path: Path, output: Path) -> Path:
    """Render training and validation curves without matplotlib."""
    batches = [row for row in _read_jsonl(Path(batch_path)) if row.get("type") == "batch"]
    epochs = _read_jsonl(Path(epoch_path))
    steps = [float(row.get("step", 0)) for row in batches]
    series = {
        "total": [float(row.get("total", math.nan)) for row in batches],
        "parent": [float(row.get("parent", math.nan)) for row in batches],
        "child": [float(row.get("child", math.nan)) for row in batches],
        "grad_norm": [float(row.get("grad_norm", math.nan)) for row in batches],
    }
    epoch_steps = [float(row.get("global_step", 0)) for row in epochs]
    val_total = [float(row.get("val", {}).get("total", math.nan)) for row in epochs]
    val_parent = [float(row.get("val", {}).get("parent", math.nan)) for row in epochs]
    val_child = [float(row.get("val", {}).get("child", math.nan)) for row in epochs]
    width, height = 1040, 620
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append('<text x="24" y="30" font-family="sans-serif" font-size="18" font-weight="700">OLM-Gold learning curves</text>')
    parts.append(_panel(40, 70, "Training total / parent / child", [("total", series["total"], "#1f77b4"), ("parent", series["parent"], "#ff7f0e"), ("child", series["child"], "#2ca02c")]))
    parts.append(_panel(540, 70, "Validation total / parent / child", [("total", val_total, "#1f77b4"), ("parent", val_parent, "#ff7f0e"), ("child", val_child, "#2ca02c")]))
    parts.append(_panel(40, 370, "Gradient norm", [("grad_norm", series["grad_norm"], "#d62728")]))
    parts.append(_panel(540, 370, "Validation NLL by epoch (same as total)", [("val_total", val_total, "#9467bd")]))
    parts.append('<text x="40" y="600" font-family="sans-serif" font-size="11">Batch points shown: ' + str(len(steps)) + " | epochs: " + str(len(epochs)) + "</text>")
    parts.append("</svg>")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")
    return output