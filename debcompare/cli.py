"""Command-line entry point for the benchmark."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from debcompare.benchmark import run_benchmark


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="debcompare",
        description=(
            "Reproducible benchmark: Classical DEB vs Bayesian DEB vs Random Forest. "
            "Companion to Hackenberger & Djerdj (2026), Ecological Modelling."
        ),
    )
    p.add_argument("-o", "--output-dir", default="benchmark_output",
                   help="Directory for CSV and figure output (default: benchmark_output)")
    p.add_argument("--draws", type=int, default=2000,
                   help="NUTS draws per chain (default: 2000)")
    p.add_argument("--tune", type=int, default=1000,
                   help="NUTS tuning steps (default: 1000)")
    p.add_argument("--chains", type=int, default=3,
                   help="NUTS chains (default: 3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the 3-panel benchmark figure (CSV only)")
    p.add_argument("--gallery", action="store_true",
                   help="Also render the 6-plot educational gallery (pydeb.plots) as PDF+PNG.")
    p.add_argument("--progressbar", action="store_true",
                   help="Show PyMC progress bar")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print informational log messages")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.output_dir).resolve()
    print(f"Running benchmark; results will be written to: {out_dir}")

    result = run_benchmark(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
        output_dir=out_dir,
        save_figure=not args.no_figure,
        progressbar=args.progressbar,
    )

    # Pretty print
    print("\n=== Benchmark results ===\n")
    with open(out_dir / "benchmark_results.csv") as f:
        print(f.read())
    print("\n=== Posterior summary ===\n")
    print(result.bayesian_posterior_summary.to_string())
    print(f"\nWrote:\n  {out_dir / 'benchmark_results.csv'}")
    if not args.no_figure:
        print(f"  {out_dir / 'benchmark_figure.pdf'}")
        print(f"  {out_dir / 'benchmark_figure.png'}")

    if args.gallery:
        from pydeb.plots import render_gallery

        gallery_dir = out_dir / "gallery"
        print(f"\nRendering educational plot gallery to: {gallery_dir}")
        written = render_gallery(result, gallery_dir, random_seed=args.seed)
        for name, paths in written.items():
            print(f"  {name}: {paths[0].name}, {paths[1].name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
