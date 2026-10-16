"""Command line entry point: `ghost preprocess | train | score`."""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from .model import ModelConfig
from .train import TrainConfig

DEFAULT_RAW = Path('data/raw/bdb2023')
DEFAULT_DATA = Path('data/processed/bdb2023.npz')
DEFAULT_RUNS = Path('runs/default')


def add_dataclass_args(parser: argparse.ArgumentParser, cls, skip=()) -> None:
    for f in fields(cls):
        if f.name in skip:
            continue
        parser.add_argument(f'--{f.name.replace("_", "-")}', type=type(f.default), default=None,
                            help=f'default: {f.default}')


def collect(args: argparse.Namespace, cls, skip=()) -> dict:
    return {f.name: getattr(args, f.name) for f in fields(cls)
            if f.name not in skip and getattr(args, f.name) is not None}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='ghost')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('preprocess', help='raw tracking CSVs -> array cache')
    p.add_argument('--raw', type=Path, default=DEFAULT_RAW)
    p.add_argument('--out', type=Path, default=DEFAULT_DATA)
    p.add_argument('--max-frames', type=int, default=50, help='frames kept after the snap')
    p.add_argument('--min-frames', type=int, default=10)

    model_required = ('n_dyn', 'n_glob', 'n_coverage', 'max_frames')
    p = sub.add_parser('train', help='train one fold (or all data with --fold all)')
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--runs', type=Path, default=DEFAULT_RUNS)
    p.add_argument('--fold', default='0', help='held-out fold index, or "all"')
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--device', default='auto')
    add_dataclass_args(p, ModelConfig, skip=model_required)
    add_dataclass_args(p, TrainConfig)

    p = sub.add_parser('score', help='heliocentricity from cross-fitted fold models')
    p.add_argument('--data', type=Path, default=None, help='default: the cache the folds trained on')
    p.add_argument('--runs', type=Path, default=DEFAULT_RUNS)
    p.add_argument('--out', type=Path, default=None, help='default: <runs>/scores')
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--players', type=Path, default=DEFAULT_RAW / 'players.csv')
    p.add_argument('--device', default='auto')

    args = parser.parse_args(argv)
    if args.command == 'preprocess':
        from .preprocess import preprocess
        preprocess(args.raw, args.out, args.max_frames, args.min_frames)
    elif args.command == 'train':
        from .train import train
        fold = None if args.fold == 'all' else int(args.fold)
        out_dir = args.runs / ('all' if fold is None else f'fold{fold}')
        train(args.data, out_dir, fold, args.n_folds,
              collect(args, ModelConfig, skip=model_required),
              TrainConfig(**collect(args, TrainConfig)), args.device)
    elif args.command == 'score':
        from .score import score
        score(args.data, args.runs, args.out or args.runs / 'scores', args.samples,
              args.batch_size, args.players, args.device)


if __name__ == '__main__':
    main()
