# FILE MAP | Standalone results collector (thin CLI over src/results_summary.py).
#   Scans local results/ and (if present) the Drive mirror for every per-run metrics.json,
#   consolidates them into results/summary/ (SUMMARY.md + CSVs + JSON). PURE stdlib, no GPU,
#   no torch — safe to run on any laptop. Re-run any time to absorb new seeds / models; it never
#   retrains or re-evaluates, it only reads what training already wrote.
"""
Usage:
  python aggregate_results.py                       # scan results/ (+ Drive mirror if mounted)
  python aggregate_results.py --results-dir results --out results/summary
  python aggregate_results.py --drive-dir /content/drive/MyDrive/msu2026_checkpoints/results
"""

import argparse
import sys
from pathlib import Path

from src.results_summary import DEFAULT_DRIVE_RESULTS, build_summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Consolidate per-run metrics.json into results/summary/.")
    p.add_argument("--results-dir", default="results",
                   help="local results root (default: results)")
    p.add_argument("--drive-dir", default=DEFAULT_DRIVE_RESULTS,
                   help="Drive results mirror to also scan if it exists (default: the Colab path)")
    p.add_argument("--out", default="results/summary",
                   help="output folder for the summary (default: results/summary)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    # Local first so local copies win over the Drive mirror for the same model/seed.
    roots = [args.results_dir, args.drive_dir]
    present = [r for r in roots if Path(r).is_dir()]
    if not present:
        print(f"No result roots found. Looked in: {roots}")
        print("Pull your Drive 'msu2026_checkpoints/results/' folder into 'results/' "
              "(or run this in Colab with Drive mounted), then re-run.")
        return 1

    print(f"Scanning: {', '.join(present)}")
    report = build_summary(roots, args.out)
    if report["n_runs"] == 0:
        print("Found the root(s) but no metrics.json inside. Nothing to summarize yet.")
        return 1

    print(f"\nFound {report['n_runs']} run(s):")
    for run in report["runs"]:
        print(f"  - {run}")
    print("\nWrote:")
    for label, path in report["written"].items():
        print(f"  {label:<22} {path}")
    print(f"\nOpen {report['written']['summary_md']} for the readable tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
