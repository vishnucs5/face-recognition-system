"""Evaluation pipeline computing biometric identification metrics across varying thresholds."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from app.config import config
from app.database import DatabaseManager
from app.recognition.recognizer import FaceRecognitionService

logger = logging.getLogger(__name__)


@dataclass
class TestSampleResult:
    """Individual test sample evaluation result."""

    image_path: str
    ground_truth: str  # Identity name or "UNKNOWN"
    is_known: bool
    faces_detected: int
    predicted_identity: Optional[str]
    best_similarity: float
    raw_status: str


@dataclass
class ThresholdMetrics:
    """Metrics evaluated at a specific similarity threshold."""

    threshold: float
    total_known: int
    total_unknown: int
    correct_known_accepts: int
    false_known_rejects: int
    misidentifications: int
    correct_unknown_rejects: int
    false_unknown_accepts: int
    tar: float  # True Accept Rate (Known correct / Total known)
    frr: float  # False Reject Rate ((Known rejects + Misidentifications) / Total known)
    far: float  # False Accept Rate (Unknown accepts / Total unknown)
    trr: float  # True Reject Rate (Unknown rejects / Total unknown)
    accuracy: float  # Overall identification accuracy


@dataclass
class EvaluationReport:
    """Consolidated evaluation summary and analysis report."""

    dataset_available: bool
    num_known_samples: int
    num_unknown_samples: int
    threshold_metrics: List[ThresholdMetrics]
    best_threshold: Optional[float]
    best_accuracy: Optional[float]
    summary_message: str


class EvaluationService:
    """Runs threshold sweeps, calculates verification metrics, and generates visual reports."""

    def __init__(
        self,
        recognition_service: Optional[FaceRecognitionService] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.recognizer = (
            recognition_service if recognition_service is not None else FaceRecognitionService()
        )
        self.output_dir = (
            Path(output_dir) if output_dir is not None else config.eval_output_dir
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scan_test_dataset(
        self, test_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[List[Tuple[Path, str]], List[Path]]:
        """Discover known and unknown test images from directory structure.

        Expected structure:
        test_dir/
            known/
                Alice/
                    img1.jpg
                Bob/
                    img2.jpg
            unknown/
                person_x/
                    img3.jpg

        Returns:
            Tuple[List[Tuple[Path, str]], List[Path]]:
                - known_samples: [(image_path, expected_name)]
                - unknown_samples: [image_path]
        """
        root = Path(test_dir) if test_dir else config.test_dir
        known_root = root / "known"
        unknown_root = root / "unknown"

        known_samples: List[Tuple[Path, str]] = []
        unknown_samples: List[Path] = []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if known_root.is_dir():
            for person_dir in sorted(known_root.iterdir()):
                if person_dir.is_dir():
                    name = person_dir.name
                    for img_file in person_dir.glob("*.*"):
                        if img_file.suffix.lower() in valid_exts:
                            known_samples.append((img_file, name))

        if unknown_root.is_dir():
            for item in unknown_root.rglob("*.*"):
                if item.is_file() and item.suffix.lower() in valid_exts:
                    unknown_samples.append(item)

        return known_samples, unknown_samples

    def evaluate(
        self,
        test_dir: Optional[Union[str, Path]] = None,
        thresholds: Optional[List[float]] = None,
    ) -> EvaluationReport:
        """Execute evaluation across the test dataset and sweep thresholds.

        Args:
            test_dir: Optional path to test directory.
            thresholds: List of similarity thresholds to test (defaults to 0.30 to 0.80).

        Returns:
            EvaluationReport: Report with detailed threshold performance metrics.
        """
        if thresholds is None:
            thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

        known_samples, unknown_samples = self.scan_test_dataset(test_dir)
        total_known = len(known_samples)
        total_unknown = len(unknown_samples)

        if total_known == 0 and total_unknown == 0:
            msg = (
                "Evaluation results cannot be confirmed until real test images are supplied. "
                "Please place labeled test images in 'data/test/known/<person_name>/' and 'data/test/unknown/'."
            )
            logger.warning(msg)
            return EvaluationReport(
                dataset_available=False,
                num_known_samples=0,
                num_unknown_samples=0,
                threshold_metrics=[],
                best_threshold=None,
                best_accuracy=None,
                summary_message=msg,
            )

        logger.info(
            "Running evaluation on %d known sample(s) and %d unknown sample(s)...",
            total_known,
            total_unknown,
        )

        # Retrieve enrolled embeddings matrix once for fast comparison
        db_matrix, person_ids, person_names = self.recognizer.db.get_all_embeddings_matrix()

        if len(db_matrix) == 0:
            msg = "Enrollment database is empty. Cannot evaluate without enrolled reference identities."
            logger.warning(msg)
            return EvaluationReport(
                dataset_available=True,
                num_known_samples=total_known,
                num_unknown_samples=total_unknown,
                threshold_metrics=[],
                best_threshold=None,
                best_accuracy=None,
                summary_message=msg,
            )

        # Collect raw embedding comparisons for all test items
        all_eval_items = []

        # 1. Process known samples
        for img_path, expected_name in known_samples:
            try:
                output = self.recognizer.identify(img_path, threshold=0.0)  # query with threshold 0 to get raw similarity
                if output.num_faces_detected > 0:
                    best_res = output.results[0]
                    sim = best_res.similarity
                    cand_name = best_res.identity
                else:
                    sim = 0.0
                    cand_name = None

                all_eval_items.append(
                    {
                        "path": str(img_path),
                        "ground_truth": expected_name,
                        "is_known": True,
                        "faces_detected": output.num_faces_detected,
                        "candidate_identity": cand_name,
                        "similarity": sim,
                    }
                )
            except Exception as e:
                logger.error("Error processing %s: %s", img_path, e)

        # 2. Process unknown samples
        for img_path in unknown_samples:
            try:
                output = self.recognizer.identify(img_path, threshold=0.0)
                if output.num_faces_detected > 0:
                    best_res = output.results[0]
                    sim = best_res.similarity
                    cand_name = best_res.identity
                else:
                    sim = 0.0
                    cand_name = None

                all_eval_items.append(
                    {
                        "path": str(img_path),
                        "ground_truth": "UNKNOWN",
                        "is_known": False,
                        "faces_detected": output.num_faces_detected,
                        "candidate_identity": cand_name,
                        "similarity": sim,
                    }
                )
            except Exception as e:
                logger.error("Error processing %s: %s", img_path, e)

        # Sweep thresholds
        threshold_metrics_list: List[ThresholdMetrics] = []

        for th in thresholds:
            correct_known = 0
            known_rejects = 0
            misidentifications = 0
            correct_unknown = 0
            unknown_accepts = 0

            for item in all_eval_items:
                sim = item["similarity"]
                cand = item["candidate_identity"]
                gt = item["ground_truth"]
                is_known = item["is_known"]

                is_accepted = sim >= th

                if is_known:
                    if is_accepted:
                        if cand and cand.lower() == gt.lower():
                            correct_known += 1
                        else:
                            misidentifications += 1
                    else:
                        known_rejects += 1
                else:
                    # Ground truth is UNKNOWN
                    if is_accepted:
                        unknown_accepts += 1
                    else:
                        correct_unknown += 1

            tar = (correct_known / total_known) if total_known > 0 else 0.0
            frr = ((known_rejects + misidentifications) / total_known) if total_known > 0 else 0.0
            far = (unknown_accepts / total_unknown) if total_unknown > 0 else 0.0
            trr = (correct_unknown / total_unknown) if total_unknown > 0 else 0.0
            total_samples = total_known + total_unknown
            acc = ((correct_known + correct_unknown) / total_samples) if total_samples > 0 else 0.0

            threshold_metrics_list.append(
                ThresholdMetrics(
                    threshold=th,
                    total_known=total_known,
                    total_unknown=total_unknown,
                    correct_known_accepts=correct_known,
                    false_known_rejects=known_rejects,
                    misidentifications=misidentifications,
                    correct_unknown_rejects=correct_unknown,
                    false_unknown_accepts=unknown_accepts,
                    tar=round(tar, 4),
                    frr=round(frr, 4),
                    far=round(far, 4),
                    trr=round(trr, 4),
                    accuracy=round(acc, 4),
                )
            )

        # Identify best threshold by overall accuracy
        best_metric = max(threshold_metrics_list, key=lambda m: m.accuracy)

        # Save output artifacts
        self._save_results(all_eval_items, threshold_metrics_list, best_metric.threshold)

        summary = (
            f"Evaluated {len(all_eval_items)} sample(s). "
            f"Optimal threshold observed: {best_metric.threshold:.2f} with Accuracy={best_metric.accuracy*100:.1f}%, "
            f"TAR={best_metric.tar*100:.1f}%, FAR={best_metric.far*100:.1f}%, FRR={best_metric.frr*100:.1f}%."
        )

        return EvaluationReport(
            dataset_available=True,
            num_known_samples=total_known,
            num_unknown_samples=total_unknown,
            threshold_metrics=threshold_metrics_list,
            best_threshold=best_metric.threshold,
            best_accuracy=best_metric.accuracy,
            summary_message=summary,
        )

    def _save_results(
        self,
        samples: List[dict],
        metrics: List[ThresholdMetrics],
        best_th: float,
    ) -> None:
        """Persist evaluation tables, metrics, plots, and JSON report."""
        # 1. Save sample-by-sample CSV
        df_samples = pd.DataFrame(samples)
        df_samples.to_csv(self.output_dir / "results.csv", index=False)

        # 2. Save threshold sweep CSV
        df_metrics = pd.DataFrame([asdict(m) for m in metrics])
        df_metrics.to_csv(self.output_dir / "threshold_analysis.csv", index=False)

        # 3. Save report JSON
        report_data = {
            "num_samples": len(samples),
            "best_threshold": best_th,
            "metrics": [asdict(m) for m in metrics],
        }
        with open(self.output_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 4. Generate visual plot: FAR, FRR, and Accuracy vs Threshold
        try:
            fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
            ths = [m.threshold for m in metrics]
            fars = [m.far * 100 for m in metrics]
            frrs = [m.frr * 100 for m in metrics]
            accs = [m.accuracy * 100 for m in metrics]

            ax.plot(ths, accs, marker="o", color="#10B981", linewidth=2.5, label="Overall Accuracy (%)")
            ax.plot(ths, fars, marker="s", color="#EF4444", linewidth=2.0, linestyle="--", label="False Accept Rate FAR (%)")
            ax.plot(ths, frrs, marker="^", color="#3B82F6", linewidth=2.0, linestyle="-.", label="False Reject Rate FRR (%)")

            ax.axvline(best_th, color="#6366F1", linestyle=":", label=f"Best Threshold ({best_th:.2f})")
            ax.set_title("Threshold Analysis: Accuracy, FAR, and FRR Tradeoff", fontsize=14, pad=12)
            ax.set_xlabel("Cosine Similarity Threshold", fontsize=11)
            ax.set_ylabel("Rate (%)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="best")
            plt.tight_layout()
            plot_path = self.output_dir / "threshold_analysis.png"
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info("Saved threshold analysis plot to %s", plot_path)
        except Exception as e:
            logger.warning("Could not generate threshold plot: %s", e)
