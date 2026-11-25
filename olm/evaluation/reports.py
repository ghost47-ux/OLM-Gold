"""Markdown/JSON evaluation reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def render_report(payload: dict) -> str:
    evaluation = payload['evaluation']
    metrics = evaluation['metrics']
    coordinates = evaluation.get('per_coordinate', {})
    lines = [
        '# OLM-Gold evaluation report',
        '',
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        '',
        f"- Checkpoint: `{payload.get('checkpoint', 'unknown')}`",
        f"- Split: `{payload.get('split', 'unknown')}` ({evaluation['n']} samples)",
        "- Mode: teacher-forced one-step (child decoder conditioned on the actual parent and preceding children)",
        '',
        '## Headline metrics',
        '',
        '| Metric | Value |',
        '| --- | --- |',
    ]
    for key in (
        'parent_mae', 'parent_normalized_mae', 'child_mae', 'child_normalized_mae',
        'body_error', 'wick_error', 'range_error', 'aggregation_log_error',
        'parent_nll_model_space', 'child_nll_model_space', 'crps_mean',
        'coverage_50_mean', 'coverage_90_mean',
    ):
        if key in metrics:
            lines.append(f'| {key} | {metrics[key]:.6f} |')
    lines += ['', '## Per-coordinate error', '', '| Coordinate | Parent MAE | Child MAE | CRPS |', '| --- | --- | --- | --- |']
    for name in ('open', 'high', 'low', 'close'):
        lines.append(
            f"| {name} | {coordinates.get('parent_mae', {}).get(name, float('nan')):.6f} "
            f"| {coordinates.get('child_mae', {}).get(name, float('nan')):.6f} "
            f"| {coordinates.get('crps', {}).get(name, float('nan')):.6f} |"
        )
    lines += ['', '## Calibration', '', '| Interval | Open | High | Low | Close |', '| --- | --- | --- | --- | --- |']
    for level in ('coverage_50', 'coverage_90'):
        row = coordinates.get(level, {})
        lines.append(
            f"| {level.replace('coverage_', '')}% | " + ' | '.join(f"{row.get(name, float('nan')):.3f}" for name in ('open', 'high', 'low', 'close')) + ' |'
        )
    if payload.get('free_running'):
        free = payload['free_running']
        lines += [
            '',
            '## Free-running (recursive) evaluation',
            '',
            f"- Child MAE: {free['child_mae_free_running']:.6f}",
            f"- Aggregate log error: {free['aggregate_log_error_free_running']:.6f}",
            f"- Predicted parent vs child aggregate: {free['predicted_parent_aggregate_abs_error']:.6f}",
        ]
    lines += [
        '',
        '## Interpretation limits',
        '',
        '- Teacher-forced metrics do not measure multi-step compounding; use the free-running evaluation for that.',
        '- Calibration coverage is estimated from the parent mixture samples and is not a guarantee of live performance.',
        '- A lower training loss is not evidence of predictive skill; these results are measured on held-out chronological data.',
        '',
    ]
    return '\n'.join(lines)


def write_report(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == '.json':
        target.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    else:
        target.write_text(render_report(payload), encoding='utf-8')
    return target
