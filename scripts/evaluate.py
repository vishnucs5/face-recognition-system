#!/usr/bin/env python3
"""CLI utility to run biometric evaluation, threshold sweeps, and generate performance plots."""

import argparse
from dataclasses import asdict
import logging
import sys
from pathlib import Path
from tabulate import tabulate

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config
from app.evaluation.evaluator import EvaluationService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_cli")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation sweep across similarity thresholds."
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(config.test_dir),
        help=f"Root directory containing 'known/' and 'unknown/' subdirectories (default: {config.test_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.eval_output_dir),
        help=f"Directory to save evaluation reports and plots (default: {config.eval_output_dir})",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        help="Comma-separated list of similarity thresholds to evaluate",
    )

    args = parser.parse_args()

    threshold_list = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    eval_service = EvaluationService(output_dir=args.output_dir)
    report = eval_service.evaluate(test_dir=args.test_dir, thresholds=threshold_list)

    print("\n" + "=" * 80)
    print("                 BIOMETRIC IDENTIFICATION EVALUATION")
    print("=" * 80)

    if not report.dataset_available or len(report.threshold_metrics) == 0:
        print(f"\nNotice: {report.summary_message}\n")
        print("=" * 80 + "\n")
        sys.exit(0)

    table_data = []
    for m in report.threshold_metrics:
        table_data.append(
            [
                f"{m.threshold:.2f}",
                m.total_known,
                m.total_unknown,
                m.correct_known_accepts,
                m.false_known_rejects,
                m.false_unknown_accepts,
                m.correct_unknown_rejects,
                f"{m.tar * 100:.1f}%",
                f"{m.far * 100:.1f}%",
                f"{m.frr * 100:.1f}%",
                f"{m.accuracy * 100:.1f}%",
            ]
        )

    headers = [
        "Threshold",
        "Known",
        "Unknown",
        "Known Acc",
        "Known Rej",
        "Unk Acc (FAR)",
        "Unk Rej (TRR)",
        "TAR",
        "FAR",
        "FRR",
        "Accuracy",
    ]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\n" + report.summary_message)
    print(f"\nArtifacts saved to: {Path(args.output_dir).resolve()}")
    print("  - results.csv (detailed per-image classifications)")
    print("  - threshold_analysis.csv (metrics table)")
    print("  - report.json (machine-readable metrics)")
    print("  - threshold_analysis.png (FAR/FRR trade-off curve)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
