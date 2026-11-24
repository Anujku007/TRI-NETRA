"""Run the shared official-protocol WIDER FACE evaluation over saved detections."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy

from widerface_evaluation import evaluate_detector, load_annotations, load_predictions, write_pr_csv


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "research-data/experiment-config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _format(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _format_interval(value: float | None, interval: dict[str, float] | None, digits: int = 4) -> str:
    if value is None or interval is None:
        return "n/a"
    return f"{value:.{digits}f} [{interval['lower']:.{digits}f}, {interval['upper']:.{digits}f}]"


def _format_pair_interval(value: float | None, interval: list[float] | None) -> str:
    if value is None or interval is None:
        return "n/a"
    return f"{value:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"


def _write_research_report(
    output_path: Path,
    summary: dict[str, Any],
    calibration: dict[str, Any] | None,
) -> None:
    lines = [
        "# TRI-Netra Research Phase 2: Experimental Evaluation",
        "",
        "This report contains measured results from the configured local evaluation runs. It does not claim improvement over prior work or state-of-the-art methods.",
        "",
        "## WIDER FACE validation",
        "",
        "Easy, Medium, and Hard use the official WIDER FACE validation keep-index annotations. Predictions overlapping non-kept boxes are ignored. Both detectors use the same image list, inclusive WIDER IoU arithmetic, score normalization, 1,000-point PR thresholds, AP integration, and image-cluster bootstrap samples.",
        "",
        "Precision, recall, and F1 are measured at each detector's predeclared native output point. AP and PR curves use all emitted detections ranked by detector-native scores. Haar level weights are rankings, not calibrated probabilities; score magnitudes are not compared between detector families.",
        "",
        "| Detector | Subset | Images | GT | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | AP (95% CI) | Mean latency ms (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for detector_name, detector in summary["detectors"].items():
        for difficulty in ("Easy", "Medium", "Hard"):
            metric = detector["difficulty"][difficulty]
            intervals = metric["confidence_interval_95"]
            latency = metric["inference_latency_ms"]["mean"]
            lines.append(
                f"| {detector_name} | {difficulty} | {detector['images_evaluated']} | "
                f"{metric['ground_truth_faces']} | {metric['true_positives']} | "
                f"{metric['false_positives']} | {metric['false_negatives']} | "
                f"{_format_interval(metric['precision'], intervals['precision'])} | "
                f"{_format_interval(metric['recall'], intervals['recall'])} | "
                f"{_format_interval(metric['f1'], intervals['f1'])} | "
                f"{_format_interval(metric['average_precision'], intervals['average_precision'])} | "
                f"{_format_interval(latency, intervals['mean_inference_latency_ms'], 3)} |"
            )
    lines.extend([
        "",
        "Intervals are percentile 95% confidence intervals from 1,000 image-cluster bootstrap replicates. PR points are in `results/widerface_pr_curves.csv`.",
        "",
        "## Identity score calibration",
        "",
    ])
    if calibration:
        lines.extend([
            f"The runtime score implementation was imported directly from `{calibration['runtimeScoring']['implementation']}`. The calibration and evaluation identities are disjoint: **{calibration['split']['identityDisjoint']}**.",
            "",
            "| Split | Usable identities | Usable images | Adequacy flag | Runtime threshold | Same-identity match (95% CI) | Different-identity false-match (95% CI) |",
            "|---|---:|---:|---|---:|---:|---:|",
        ])
        for split_name in ("calibration", "evaluation"):
            split = calibration[split_name]
            runtime_row = next(
                (row for row in split["thresholdResults"] if row["isRuntimeAuthorizationThreshold"]),
                None,
            )
            if runtime_row:
                lines.append(
                    f"| {split_name} | {split['identityCount']} | {split['usableImageCount']} | "
                    f"{split['sufficientForReliableCalibration']} | {runtime_row['threshold']:.2f} | "
                    f"{_format_pair_interval(runtime_row['samePersonMatchRate'], runtime_row['identityClusterBootstrap95CI']['samePersonMatchRate'])} | "
                    f"{_format_pair_interval(runtime_row['differentPersonFalseMatchRate'], runtime_row['identityClusterBootstrap95CI']['differentPersonFalseMatchRate'])} |"
                )
        lines.extend([
            "",
            f"Coarse-hash pairs flagged for review: {calibration['duplicateAudit']['coarseHashCandidatePairCount']} "
            f"({calibration['duplicateAudit']['coarseHashCrossIdentityCandidatePairCount']} cross-identity). "
            "Coarse-hash candidates are never discarded. Only same-identity exact-byte duplicates are removed.",
            "",
            f"Exact-byte duplicate identity groups kept within one split: {calibration['split']['crossIdentityExactByteGroupsKeptTogether']}.",
            "",
            "Runtime gallery identification uses one deterministic enrollment image per identity and all remaining images as queries. It applies both the runtime authorization threshold and best-versus-runner-up margin; its confidence intervals resample identities.",
            "",
            "| Split | Gallery images | Queries | Rank-1 accuracy (95% CI) | Accepted coverage (95% CI) | Accepted accuracy (95% CI) | False accepts/query (95% CI) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for split_name in ("calibration", "evaluation"):
            identification = calibration[split_name]["runtimeIdentification"]
            lines.append(
                f"| {split_name} | {identification['galleryImages']} | {identification['queryCount']} | "
                f"{_format_pair_interval(identification['rankOneAccuracy'], identification['identityClusterBootstrap95CI']['rankOneAccuracy'])} | "
                f"{_format_pair_interval(identification['acceptedCoverage'], identification['identityClusterBootstrap95CI']['acceptedCoverage'])} | "
                f"{_format_pair_interval(identification['acceptedAccuracy'], identification['identityClusterBootstrap95CI']['acceptedAccuracy'])} | "
                f"{_format_pair_interval(identification['falseAcceptRatePerQuery'], identification['identityClusterBootstrap95CI']['falseAcceptRatePerQuery'])} |"
            )
    else:
        lines.append("Identity calibration results were not found; run the configured calibration command to populate this section.")
    lines.extend([
        "",
        "## Reproducibility and limitations",
        "",
        f"- Experiment configuration SHA-256: `{summary['reproducibility']['experiment_config_sha256']}`.",
        f"- WIDER validation images scored: {summary['dataset']['images']}.",
        f"- WIDER annotation bounding boxes: {summary['dataset']['annotated_faces']}.",
        "- Human.js uses the unchanged runtime detector service, including its native confidence and maximum detections per image. Haar uses OpenCV's default configured cascade and native level weights.",
        "- Results are conditional on the bundled LFW sample, local model artifacts, and the current CPU/software environment. They are not independent external validation of biometric identity accuracy or of production performance.",
        "- Confidence intervals resample images for detection metrics and identities for identity-pair calibration; they do not account for dataset shift or annotation error.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_config()
    wider = config["widerFace"]
    output_paths = wider["outputs"]
    annotation_path = ROOT / wider["annotationMat"]
    difficulty_paths = {name: ROOT / path for name, path in wider["difficultyAnnotations"].items()}
    annotations = load_annotations(annotation_path, difficulty_paths)
    expected_count = wider.get("expectedImageCount")
    if expected_count is not None and len(annotations) != expected_count:
        raise ValueError(f"Expected {expected_count} WIDER validation images, loaded {len(annotations)}")

    detector_reports: dict[str, Any] = {}
    curves_by_detector: dict[str, dict[str, list[dict[str, float]]]] = {}
    prediction_hashes: dict[str, str] = {}
    for detector_name in ("humanjs", "haar"):
        prediction_path = ROOT / wider["predictionFiles"][detector_name]
        metadata_path = ROOT / wider["metadataFiles"][detector_name]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        predictions, inference_ms = load_predictions(prediction_path, annotations)
        report, curves = evaluate_detector(
            annotations=annotations,
            predictions_by_image=predictions,
            inference_ms_by_image=inference_ms,
            detector_metadata=metadata,
            iou_threshold=float(wider["iouThreshold"]),
            threshold_count=int(wider["scoreThresholdCount"]),
            bootstrap_replicates=int(wider["bootstrapReplicates"]),
            bootstrap_seed=int(wider["bootstrapSeed"]),
        )
        detector_reports[detector_name] = report
        curves_by_detector[detector_name] = curves
        prediction_hashes[detector_name] = sha256(prediction_path)

    config_hash = sha256(CONFIG_PATH)
    calibration_path = ROOT / "research-data/results/identity_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else None
    summary = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "dataset": {
            "name": "WIDER FACE validation",
            "images": len(annotations),
            "annotated_faces": int(sum(len(annotation.boxes) for annotation in annotations)),
            "annotation_mat_sha256": sha256(annotation_path),
            "difficulty_annotations_sha256": {
                name: sha256(path) for name, path in difficulty_paths.items()
            },
        },
        "protocol": {
            **wider["protocol"],
            "iou_threshold": float(wider["iouThreshold"]),
            "score_threshold_count": int(wider["scoreThresholdCount"]),
            "bootstrap_replicates": int(wider["bootstrapReplicates"]),
            "bootstrap_seed": int(wider["bootstrapSeed"]),
        },
        "reproducibility": {
            "experiment_config_sha256": config_hash,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "scipy_version": scipy.__version__,
            "prediction_file_sha256": prediction_hashes,
        },
        "detectors": detector_reports,
        "identity_calibration": calibration,
    }

    summary_path = ROOT / output_paths["summary"]
    pr_path = ROOT / output_paths["prCurvesCsv"]
    report_path = ROOT / output_paths["reportMarkdown"]
    for output_path in (summary_path, pr_path, report_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_pr_csv(pr_path, curves_by_detector)

    for detector_name, legacy_path_key in (
        ("humanjs", "legacyHumanjsReport"),
        ("haar", "legacyHaarReport"),
    ):
        detector = detector_reports[detector_name]
        overall = detector["difficulty"]["All"]
        legacy = {
            "schema_version": 2,
            "detector": detector["detector"].get("detector", detector_name),
            "dataset": "WIDER FACE validation",
            "iou_threshold": float(wider["iouThreshold"]),
            "images_evaluated": detector["images_evaluated"],
            "ground_truth_faces": overall["ground_truth_faces"],
            "predicted_faces": detector["raw_predictions"],
            "TP": overall["true_positives"],
            "FP": overall["false_positives"],
            "FN": overall["false_negatives"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1_score": overall["f1"],
            "average_precision": overall["average_precision"],
            "confidence_interval_95": overall["confidence_interval_95"],
            "inference_latency_ms": overall["inference_latency_ms"],
            "difficulty_results": detector["difficulty"],
            "reproducibility": summary["reproducibility"],
        }
        legacy_path = ROOT / output_paths[legacy_path_key]
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(json.dumps(legacy, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    _write_research_report(report_path, summary, calibration)
    print(f"WIDER FACE validation images: {len(annotations)}")
    for detector_name, detector in detector_reports.items():
        print(detector_name)
        for difficulty in ("Easy", "Medium", "Hard"):
            metrics = detector["difficulty"][difficulty]
            print(
                f"  {difficulty}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} AP={metrics['average_precision']:.4f} "
                f"mean latency={metrics['inference_latency_ms']['mean']:.3f} ms"
            )
    print(f"Summary: {summary_path.relative_to(ROOT)}")
    print(f"PR curves: {pr_path.relative_to(ROOT)}")
    print(f"Report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
