"""Command line interface: `python -m olm <command>`."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from olm import __version__
from olm.config import Config, load_config
from olm.data.datasets import leakage_check
from olm.data.loader import load_market_frames
from olm.data.preparation import prepare_datasets
from olm.data.providers.twelvedata import TwelveDataClient
from olm.data.credentials import AuthorizationGates, RequestScheduler, build_scheduler, discover_credentials
from olm.data.ingestion import IngestionService, summarize_dataset
from olm.data.storage import DataPaths
from olm.data.validation import validate_frame
from olm.evaluation.metrics import collect_predictions, evaluate_predictions
from olm.evaluation.blocks import evaluate_sequential_blocks
from olm.evaluation.free_running import evaluate_free_running
from olm.evaluation.reports import render_report, write_report
from olm.experiments.tracker import ChampionRegistry, ExperimentTracker
from olm.hardware import detect_environment, resolve_device
from olm.inference.forecast import Forecaster
from olm.logging_utils import get_logger, setup_logging
from olm.models.olm import build_model
from olm.training.checkpointing import load_checkpoint, model_from_checkpoint, normalizer_from_checkpoint
from olm.training.trainer import Trainer

logger = get_logger(__name__)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--config', action='append', default=None, help='YAML config file (repeatable)')
    parser.add_argument('--data-root', default=None)


def _config(args) -> Config:
    config = load_config(args.config)
    if getattr(args, 'data_root', None):
        config.data.data_root = args.data_root
    return config


def _print(payload, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(json.dumps(payload, indent=2, default=str))


def cmd_verify_environment(args) -> int:
    config = _config(args)
    report = detect_environment()
    packages = {}
    for name in ('torch', 'numpy', 'pandas', 'pyarrow', 'duckdb', 'fastapi', 'pydantic', 'PyYAML'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    paths = DataPaths(Path(config.data.data_root))
    writable = True
    try:
        paths.ensure()
        probe = paths.root / '.write_probe'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
    except OSError:
        writable = False
    payload = {
        'olm_version': __version__,
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'machine': platform.machine(),
        'packages': packages,
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'cuda_available': torch.cuda.is_available(),
        'environment': report.as_dict(),
        'device': resolve_device(config.training.device),
        'data_root': str(paths.root.resolve()),
        'data_root_writable': writable,
        'disk_free_gb': round(shutil.disk_usage(paths.root.resolve()).free / 1e9, 2),
    }
    _print(payload, args.json)
    return 0


def cmd_verify_provider(args) -> int:
    gates = AuthorizationGates.from_env()
    credentials = discover_credentials()
    if not credentials:
        print('no Twelve Data credentials found; set TWELVE_DATA_API_KEY_1..7 in .env', file=sys.stderr)
        return 2
    pool = build_scheduler(gates)
    result = {
        'credentials_found': len(credentials),
        'labels': [c.label for c in credentials],
        'multi_account_pool_authorized': gates.multi_account_pool,
        'usage': pool.usage_report(),
    }
    if args.probe:
        probes = []
        targets = credentials if args.all else credentials[:1]
        for credential in targets:
            scheduler = RequestScheduler([credential], pooling_enabled=False)
            client = TwelveDataClient(scheduler)
            try:
                usage = client.api_usage()
                search = client.symbol_search(args.symbol)
                earliest = client.earliest_timestamp(args.symbol, args.interval)
                sample = client.time_series(args.symbol, args.interval, outputsize=5)
                probes.append({
                    'account': credential.label,
                    'usage': usage,
                    'instrument_matches': sum(1 for item in search if str(item.get('symbol', '')).upper() == args.symbol.upper()),
                    'earliest': earliest,
                    'sample_rows': len(sample.frame),
                    'sample_columns': list(sample.frame.columns),
                    'sample_meta': sample.meta,
                })
            except Exception as exc:
                probes.append({'account': credential.label, 'error': str(exc)[:300]})
            finally:
                client.close()
        result['probes'] = probes
    _print(result, args.json)
    return 0


def cmd_inspect_data(args) -> int:
    config = _config(args)
    frames, info = load_market_frames(config, source=args.source)
    reports = {}
    expected = {'5min': 5, '15min': 15, '1h': 60, '4h': 240}
    for interval, frame in frames.items():
        report = validate_frame(frame, expected[interval])
        reports[interval] = {'ok': report.ok, 'stats': report.stats, 'errors': report.errors, 'warnings': report.warnings}
    payload = {'info': info, 'validation': reports}
    _print(payload, args.json)
    return 0 if all(report['ok'] for report in reports.values()) else 1


def cmd_download_history(args) -> int:
    config = _config(args)
    gates = AuthorizationGates.from_env()
    if not args.dry_run:
        gates.require('data_retention', 'historical data retention')
    if args.dry_run:
        service = IngestionService(None, data_root=config.data.data_root)
        plans = {
            interval: service.plan(
                symbol=args.symbol,
                interval=interval,
                start=pd.Timestamp(args.start, tz='UTC') if args.start else None,
                end=pd.Timestamp(args.end, tz='UTC') if args.end else None,
                max_windows=args.max_windows,
                resume=not args.no_resume,
            )
            for interval in args.intervals
        }
        _print({'dry_run': True, 'retention_authorized': gates.data_retention, 'plans': plans}, args.json)
        return 0
    scheduler = build_scheduler(gates)
    client = TwelveDataClient(scheduler)
    service = IngestionService(client, data_root=config.data.data_root)
    results = {}
    try:
        for interval in args.intervals:
            results[interval] = service.download(
                symbol=args.symbol,
                interval=interval,
                start=pd.Timestamp(args.start, tz='UTC') if args.start else None,
                end=pd.Timestamp(args.end, tz='UTC') if args.end else None,
                gates=gates,
                resume=not args.no_resume,
                max_windows=args.max_windows,
            )
    finally:
        client.close()
    _print({'results': results, 'usage': scheduler.usage_report()}, args.json)
    return 0


def cmd_prepare_datasets(args) -> int:
    config = _config(args)
    bundle, manifest = prepare_datasets(
        config,
        source=args.source,
        synthetic_days=args.synthetic_days,
        force_synthetic=args.force_synthetic,
        validate=not args.skip_validation,
    )
    payload = {
        'source': manifest['source'],
        'splits': {'train': len(bundle.train), 'val': len(bundle.val), 'test': len(bundle.test)},
        'boundaries': bundle.boundaries,
        'meta': bundle.meta,
        'leakage': leakage_check(bundle),
        'normalizer': bundle.normalizer.to_dict(),
    }
    _print(payload, args.json)
    return 0


def cmd_validate_data(args) -> int:
    config = _config(args)
    frames, info = load_market_frames(config, source=args.source)
    expected = {'5min': 5, '15min': 15, '1h': 60, '4h': 240}
    ok = True
    for interval, frame in frames.items():
        report = validate_frame(frame, expected[interval])
        print(report.summary())
        ok = ok and report.ok
    return 0 if ok else 1


def _training_gate(source: str, info: dict) -> None:
    if info.get('source') == 'synthetic':
        return
    AuthorizationGates.from_env().require('model_training', 'model training on provider data')


def cmd_train(args) -> int:
    config = _config(args)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    if args.no_context:
        config.model.use_context = False
    if args.device:
        config.training.device = args.device
    frames, info = load_market_frames(config, source=args.source)
    _training_gate(info.get('source', ''), info)
    bundle, manifest = prepare_datasets(config, source=args.source, validate=not args.skip_validation)
    tracker = ExperimentTracker(config.training.run_dir)
    run_dir = tracker.start_run(args.run_name, params={'config': config.to_dict(), 'source': manifest['source']})
    trainer = Trainer(config=config, bundle=bundle, run_name=run_dir.name)
    summary = trainer.fit(resume_from=args.resume, max_steps=args.max_steps or None)
    tracker.log_metrics(run_dir, {'best_val': summary['best_val'], 'parameters': summary['parameters']})
    tracker.finish_run(run_dir, status='completed', summary=summary)
    _print({'summary': summary, 'run_dir': str(run_dir), 'checkpoints': str(trainer.checkpoint_dir)}, args.json)
    return 0


def _load_eval_bundle(config: Config, source: str):
    bundle, manifest = prepare_datasets(config, source=source, validate=True)
    return bundle, manifest


def cmd_evaluate(args) -> int:
    config = _config(args)
    checkpoint = load_checkpoint(args.checkpoint)
    bundle, manifest = _load_eval_bundle(config, args.source)
    expected = normalizer_from_checkpoint(checkpoint)
    if not (np.allclose(expected.mean, bundle.normalizer.mean, atol=1e-8) and np.allclose(expected.std, bundle.normalizer.std, atol=1e-8)):
        raise ValueError('checkpoint normalizer does not match the rebuilt dataset; refusing to evaluate')
    model = model_from_checkpoint(checkpoint)
    split = bundle.test if args.split == 'test' else bundle.val
    from torch.utils.data import DataLoader

    loader = DataLoader(split, batch_size=config.training.eval_batch_size, shuffle=False)
    data = collect_predictions(model, loader, config, 'cpu', bundle.normalizer, n_samples=args.samples)
    evaluation = evaluate_predictions(data)
    free_running = evaluate_free_running(model, loader, config, 'cpu', bundle.normalizer)
    payload = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'split': args.split,
        'manifest_source': manifest['source'],
        'evaluation': evaluation.to_dict(),
        'free_running': free_running,
        'teacher_forced': True,
        'warning': 'Metrics condition the child decoder on the actual M15 parent and actual preceding children (teacher forcing). Scenario quality needs the separate free-running evaluation.',
    }
    if args.folds > 1:
        payload['sequential_blocks'] = evaluate_sequential_blocks(
            model, split, config, 'cpu', bundle.normalizer, folds=args.folds, n_samples=max(4, args.samples // 2)
        )
    if args.report:
        write_report(args.report, payload)
        payload['report'] = str(Path(args.report).resolve())
    _print(payload, args.json)
    return 0


def cmd_benchmark(args) -> int:
    config = _config(args)
    bundle, _ = prepare_datasets(config, source=args.source, validate=False)
    model = build_model(config.model, bundle.meta['temporal_dim'])
    device = resolve_device(config.training.device)
    model.to(device)
    item = bundle.train[0]
    batch = {key: value.unsqueeze(0).repeat(args.batch_size, *([1] * value.dim())).to(device) for key, value in item.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    model.train()
    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        from olm.training.losses import compute_losses

        losses = compute_losses(model(batch), batch, config, bundle.normalizer)
        losses.total.backward()
        optimizer.step()
    start = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model(batch), batch, config, bundle.normalizer)
        losses.total.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        model.eval()
        start_inf = time.perf_counter()
        for _ in range(args.steps):
            model(batch)
        inference = time.perf_counter() - start_inf
    payload = {
        'parameters': model.parameter_count(),
        'device': device,
        'batch_size': args.batch_size,
        'train_ms_per_step': round(1000 * elapsed / args.steps, 2),
        'train_samples_per_second': round(args.batch_size * args.steps / elapsed, 2),
        'forward_ms_per_batch': round(1000 * inference / args.steps, 2),
        'threads': torch.get_num_threads(),
    }
    _print(payload, args.json)
    return 0


def cmd_forecast(args) -> int:
    config = _config(args)
    checkpoint = args.checkpoint or config.inference.checkpoint
    if args.m5_csv:
        frame = pd.read_csv(args.m5_csv)
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True)
        from olm.inference.forecast import derive_context_frames

        frames = derive_context_frames(frame)
    else:
        frames, _ = load_market_frames(config, source=args.source)
    forecaster = Forecaster(checkpoint, config)
    result = forecaster.forecast(frames, scenarios=args.scenarios, seed=args.seed)
    payload = result.to_dict()
    payload['model_version'] = forecaster.model_version
    if args.log:
        from olm.monitoring.prediction_log import PredictionLog

        log = PredictionLog(config.inference.prediction_log)
        record = log.record_forecast(payload, forecaster.model_version, result.as_of)
        payload['forecast_id'] = record['forecast_id']
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_usage(args) -> int:
    scheduler = build_scheduler(AuthorizationGates.from_env())
    _print({'usage': scheduler.usage_report()}, args.json)
    return 0


def cmd_experiment(args) -> int:
    tracker = ExperimentTracker('artifacts/runs')
    registry = ChampionRegistry('artifacts/registry.json')
    if args.action == 'list':
        _print({'runs': tracker.list_runs()}, args.json)
        return 0
    if args.action == 'promote':
        metrics_path = Path(args.metrics)
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        accepted, reason = registry.promote(
            {
                'checkpoint': str(Path(args.checkpoint).resolve()),
                'metrics': metrics['evaluation']['metrics'],
                'n_test': metrics['evaluation']['n'],
            },
            {'required_metrics': ['parent_mae', 'child_mae'], 'primary_metric': args.primary, 'min_improvement': args.min_improvement, 'min_test_samples': args.min_samples},
        )
        _print({'accepted': accepted, 'reason': reason, 'registry': str(registry.path)}, args.json)
        return 0 if accepted else 1
    if args.action == 'show':
        _print(registry.load(), args.json)
        return 0
    return 2


def cmd_serve(args) -> int:
    import uvicorn

    config = _config(args)
    os.environ['OLM_CHECKPOINT'] = str(Path(args.checkpoint or config.inference.checkpoint).resolve())
    uvicorn.run('olm.inference.api:app', host=args.host or config.inference.host, port=args.port or config.inference.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='olm', description='OLM-Gold: probabilistic XAU/USD OHLC foundation model')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('verify-environment')
    _add_common(p); p.add_argument('--json', action='store_true'); p.set_defaults(func=cmd_verify_environment)

    p = sub.add_parser('verify-provider')
    _add_common(p); p.add_argument('--probe', action='store_true'); p.add_argument('--all', action='store_true')
    p.add_argument('--symbol', default='XAU/USD'); p.add_argument('--interval', default='5min'); p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_verify_provider)

    p = sub.add_parser('inspect-data')
    _add_common(p); p.add_argument('--source', default='auto', choices=['auto', 'synthetic', 'twelvedata', 'm1']); p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_inspect_data)

    p = sub.add_parser('download-history')
    _add_common(p); p.add_argument('--symbol', default='XAU/USD')
    p.add_argument('--intervals', nargs='+', default=['5min'], choices=['1min', '5min', '15min', '1h', '4h'])
    p.add_argument('--start', default=None); p.add_argument('--end', default=None)
    p.add_argument('--max-windows', type=int, default=None); p.add_argument('--no-resume', action='store_true'); p.add_argument('--dry-run', action='store_true'); p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_download_history)

    p = sub.add_parser('prepare-datasets')
    _add_common(p); p.add_argument('--source', default='auto', choices=['auto', 'synthetic', 'twelvedata', 'm1'])
    p.add_argument('--synthetic-days', type=int, default=60); p.add_argument('--force-synthetic', action='store_true'); p.add_argument('--skip-validation', action='store_true'); p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_prepare_datasets)

    p = sub.add_parser('validate-data')
    _add_common(p); p.add_argument('--source', default='auto', choices=['auto', 'synthetic', 'twelvedata', 'm1'])
    p.set_defaults(func=cmd_validate_data)

    p = sub.add_parser('train')
    _add_common(p); p.add_argument('--source', default='auto', choices=['auto', 'synthetic', 'twelvedata'])
    p.add_argument('--run-name', default='olm'); p.add_argument('--resume', default=None)
    p.add_argument('--epochs', type=int, default=None); p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None); p.add_argument('--lr', type=float, default=None)
    p.add_argument('--no-context', action='store_true'); p.add_argument('--device', default=None)
    p.add_argument('--skip-validation', action='store_true'); p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('evaluate')
    _add_common(p); p.add_argument('--checkpoint', required=True); p.add_argument('--source', default='auto', choices=['auto', 'synthetic', 'twelvedata'])
    p.add_argument('--split', default='test', choices=['val', 'test']); p.add_argument('--samples', type=int, default=32)
    p.add_argument('--report', default=None); p.add_argument('--folds', type=int, default=1); p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser('benchmark')
    _add_common(p); p.add_argument('--source', default='auto', choices=['auto', 'synthetic', 'twelvedata'])
    p.add_argument('--batch-size', type=int, default=32); p.add_argument('--steps', type=int, default=5); p.add_argument('--warmup', type=int, default=2); p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser('forecast')
    _add_common(p); p.add_argument('--checkpoint', default=None); p.add_argument('--source', default='auto', choices=['auto', 'synthetic', 'twelvedata'])
    p.add_argument('--m5-csv', default=None); p.add_argument('--scenarios', type=int, default=32); p.add_argument('--seed', type=int, default=None)
    p.add_argument('--log', action='store_true'); p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser('serve')
    _add_common(p); p.add_argument('--checkpoint', default=None); p.add_argument('--host', default=None); p.add_argument('--port', type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser('usage')
    _add_common(p); p.add_argument('--json', action='store_true'); p.set_defaults(func=cmd_usage)

    p = sub.add_parser('experiment')
    _add_common(p); p.add_argument('action', choices=['list', 'promote', 'show'])
    p.add_argument('--checkpoint'); p.add_argument('--metrics'); p.add_argument('--primary', default='total'); p.add_argument('--min-improvement', type=float, default=0.0); p.add_argument('--min-samples', type=int, default=30); p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:
        pass
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (PermissionError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
