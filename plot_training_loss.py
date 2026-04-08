#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


LOG_LINE_RE = re.compile(
    r"^step\s+(?P<step>\d+)\s+\|\s+train\s+(?P<train>[-+]?\d*\.?\d+)\s+\|\s+val\s+(?P<val>[-+]?\d*\.?\d+)"
)
FILE_STAMP_RE = re.compile(r"(\d{8}_\d{6})")


def parse_log(log_path: Path):
    steps = []
    train_losses = []
    val_losses = []

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            match = LOG_LINE_RE.match(line)
            if not match:
                continue

            steps.append(int(match.group("step")))
            train_losses.append(float(match.group("train")))
            val_losses.append(float(match.group("val")))

    return steps, train_losses, val_losses


def collect_log_files(paths: Iterable[Path]):
    log_files = []

    for path in paths:
        if path.is_dir():
            for child in path.iterdir():
                if child.is_file() and child.suffix == ".txt" and child.name.startswith("train_"):
                    log_files.append(child)
        else:
            log_files.append(path)

    def sort_key(path: Path):
        match = FILE_STAMP_RE.search(path.name)
        if match:
            return (0, match.group(1), path.name)

        try:
            return (1, path.stat().st_mtime, path.name)
        except OSError:
            return (2, path.name)

    unique_files = []
    seen = set()
    for path in sorted(log_files, key=sort_key):
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(path)

    return unique_files


def build_parser():
    parser = argparse.ArgumentParser(
        description="Izvuci train/val loss iz logova i nacrtaj graf po koracima."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="Putanja do jednog ili vise training log fajlova.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/loss_plot.png"),
        help="Putanja izlazne slike (default: output/loss_plot.png).",
    )
    parser.add_argument(
        "--title",
        default="Train/Val Loss vs Step",
        help="Naslov grafika.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Prikazi grafik u prozoru pored snimanja u fajl.",
    )
    parser.add_argument(
        "--skip-first-steps",
        type=int,
        default=0,
        help="Preskoci prvih N koraka na grafiku.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    log_files = collect_log_files(args.logs)

    plt.figure(figsize=(12, 6))
    plotted_any = False
    step_offset = 0

    for log_path in log_files:
        if not log_path.exists():
            print(f"[WARN] Fajl ne postoji: {log_path}")
            continue

        steps, train_losses, val_losses = parse_log(log_path)
        if not steps:
            print(f"[WARN] Nema parsiranih step/train/val linija u: {log_path}")
            continue

        label_prefix = log_path.stem
        first_step = steps[0]
        adjusted_steps = [step_offset + (step - first_step) for step in steps]

        plt.plot(adjusted_steps, train_losses, label=f"{label_prefix} train", linewidth=1.5)
        plt.plot(adjusted_steps, val_losses, label=f"{label_prefix} val", linewidth=1.5, linestyle="--")
        step_offset = adjusted_steps[-1] + 1
        plotted_any = True

    if not plotted_any:
        raise SystemExit("Nema validnih podataka za crtanje.")

    if args.skip_first_steps > 0:
        plt.xlim(left=args.skip_first_steps)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.grid(True, alpha=0.25)
    #plt.legend()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Sacuvan grafik: {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
