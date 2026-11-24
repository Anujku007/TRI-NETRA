"""Shared WIDER FACE validation scoring for Human.js and OpenCV Haar."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class Annotation:
    image_id: str
    boxes: np.ndarray
    keep_indices: dict[str, set[int]]


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float


def _mat_string(value: Any) -> str:
    array = np.asarray(value)
    while array.dtype == object and array.size == 1:
        array = np.asarray(array.reshape(-1)[0])
    if array.dtype.kind in "US":
        flat = array.reshape(-1, order="F")
        if flat.size == 1:
            return str(flat[0]).strip()
        return "".join(str(part) for part in flat).strip()
    if array.size == 1:
        return str(array.reshape(-1)[0]).strip()
    raise ValueError(f"Expected a MATLAB string cell, received shape={array.shape} dtype={array.dtype}")


def _event_cells(data: dict[str, Any], key: str, event_index: int) -> list[Any]:
    event_value = data[key][event_index, 0]
    return list(np.asarray(event_value, dtype=object).reshape(-1, order="F"))


def _event_names(data: dict[str, Any]) -> list[str]:
    return [_mat_string(data["event_list"][index, 0]) for index in range(data["event_list"].shape[0])]


def _file_names(data: dict[str, Any], event_index: int) -> list[str]:
    return [_mat_string(cell) for cell in _event_cells(data, "file_list", event_index)]


def _boxes(data: dict[str, Any], event_index: int) -> list[np.ndarray]:
    result = []
    for cell in _event_cells(data, "face_bbx_list", event_index):
        array = np.asarray(cell, dtype=np.float64)
        if array.size == 0:
            result.append(np.empty((0, 4), dtype=np.float64))
            continue
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] < 4:
            raise ValueError(f"Invalid WIDER bounding box matrix: {array.shape}")
        result.append(array[:, :4])
    return result


def _keep_indices(data: dict[str, Any], event_index: int, box_counts: list[int]) -> list[set[int]]:
    event_cells = _event_cells(data, "gt_list", event_index)
    if len(event_cells) != len(box_counts):
        raise ValueError(f"WIDER subset event has {len(event_cells)} keep-index cells for {len(box_counts)} images")
    result = []
    for image_index, cell in enumerate(event_cells):
        values = np.asarray(cell, dtype=np.int64).reshape(-1)
        indices = {int(index) - 1 for index in values}
        if any(index < 0 or index >= box_counts[image_index] for index in indices):
            raise ValueError(f"WIDER keep index is outside image {image_index}'s {box_counts[image_index]} boxes")
        result.append(indices)
    return result


def load_annotations(
    annotation_mat_path: str | Path,
    difficulty_mat_paths: dict[str, str | Path],
) -> list[Annotation]:
    base = loadmat(annotation_mat_path)
    base_events = _event_names(base)
    difficulty_data = {name: loadmat(path) for name, path in difficulty_mat_paths.items()}
    difficulty_events = {name: _event_names(data) for name, data in difficulty_data.items()}
    for name, events in difficulty_events.items():
        if events != base_events:
            raise ValueError(f"WIDER {name} event order does not match base annotations")

    result: list[Annotation] = []
    for event_index, event_name in enumerate(base_events):
        names = _file_names(base, event_index)
        boxes_by_image = _boxes(base, event_index)
        if len(names) != len(boxes_by_image):
            raise ValueError(f"WIDER event {event_name} filename and box counts differ")
        keep_by_difficulty = {
            difficulty: _keep_indices(data, event_index, [len(boxes) for boxes in boxes_by_image])
            for difficulty, data in difficulty_data.items()
        }

        for image_index, name in enumerate(names):
            keep = {
                difficulty: keep_by_difficulty[difficulty][image_index]
                for difficulty in difficulty_data
            }
            keep["All"] = set(range(len(boxes_by_image[image_index])))
            result.append(Annotation(
                image_id=f"{event_name}/{name}.jpg",
                boxes=boxes_by_image[image_index],
                keep_indices=keep,
            ))
    if not result:
        raise ValueError("No WIDER validation annotations were loaded")
    return result


def inclusive_iou(first: tuple[float, float, float, float], second: np.ndarray) -> np.ndarray:
    """Mirror the official WIDER boxoverlap.m arithmetic (x+w then inclusive +1)."""
    if second.size == 0:
        return np.empty((0,), dtype=np.float64)
    x, y, width, height = first
    left = np.maximum(second[:, 0], x)
    top = np.maximum(second[:, 1], y)
    right = np.minimum(second[:, 0] + second[:, 2], x + width)
    bottom = np.minimum(second[:, 1] + second[:, 3], y + height)
    intersection_width = right - left + 1
    intersection_height = bottom - top + 1
    intersection = np.maximum(0, intersection_width) * np.maximum(0, intersection_height)
    first_area = (width + 1) * (height + 1)
    second_area = (second[:, 2] + 1) * (second[:, 3] + 1)
    union = first_area + second_area - intersection
    overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    overlap[(intersection_width <= 0) | (intersection_height <= 0)] = 0
    return overlap


def classify_image_detections(
    annotation: Annotation,
    detections: list[Detection],
    difficulty: str,
    iou_threshold: float,
) -> tuple[list[tuple[float, bool, bool, bool]], int]:
    """Return score-sorted (score, TP, FP, ignored) labels and positive GT count."""
    keep = annotation.keep_indices[difficulty]
    matched_positive: set[int] = set()
    ordered = sorted(enumerate(detections), key=lambda item: (-item[1].score, item[0]))
    labels: list[tuple[float, bool, bool, bool]] = []
    for _, detection in ordered:
        overlap = inclusive_iou(detection.bbox, annotation.boxes)
        if overlap.size == 0:
            labels.append((detection.score, False, True, False))
            continue
        matched_index = int(np.argmax(overlap))
        if overlap[matched_index] < iou_threshold:
            labels.append((detection.score, False, True, False))
        elif matched_index not in keep:
            labels.append((detection.score, False, False, True))
        elif matched_index not in matched_positive:
            matched_positive.add(matched_index)
            labels.append((detection.score, True, False, False))
        else:
            labels.append((detection.score, False, True, False))
    return labels, len(keep)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(precision.size - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def _curve_from_group_counts(
    group_scores: np.ndarray,
    group_tp: np.ndarray,
    group_fp: np.ndarray,
    ground_truth_count: int,
    threshold_count: int,
) -> tuple[list[dict[str, float]], float]:
    thresholds = np.asarray([1.0 - index / threshold_count for index in range(1, threshold_count + 1)])
    cumulative_tp = np.cumsum(group_tp)
    cumulative_fp = np.cumsum(group_fp)
    precision_points = []
    recall_points = []
    curve = []
    for threshold in thresholds:
        group_count = int(np.searchsorted(-group_scores, -threshold, side="right"))
        true_positives = float(cumulative_tp[group_count - 1]) if group_count else 0.0
        false_positives = float(cumulative_fp[group_count - 1]) if group_count else 0.0
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
        recall = true_positives / ground_truth_count if ground_truth_count else 0.0
        precision_points.append(precision)
        recall_points.append(recall)
        curve.append({"score_threshold": float(threshold), "precision": precision, "recall": recall})
    return curve, _voc_ap(np.asarray(recall_points), np.asarray(precision_points))


def _confidence_intervals(
    labels_by_image: list[list[tuple[float, bool, bool, bool]]],
    ground_truth_by_image: np.ndarray,
    inference_ms: np.ndarray,
    seed: int,
    bootstrap_replicates: int,
    threshold_count: int,
) -> dict[str, dict[str, float] | None]:
    image_count = len(labels_by_image)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, image_count, size=(bootstrap_replicates, image_count), dtype=np.int32)
    image_tp = np.asarray([sum(int(row[1]) for row in rows) for rows in labels_by_image], dtype=np.float64)
    image_fp = np.asarray([sum(int(row[2]) for row in rows) for rows in labels_by_image], dtype=np.float64)

    score_values: list[float] = []
    tp_values: list[int] = []
    fp_values: list[int] = []
    prediction_image_indices: list[int] = []
    for image_index, rows in enumerate(labels_by_image):
        for score, is_tp, is_fp, is_ignored in rows:
            if is_ignored:
                continue
            score_values.append(score)
            tp_values.append(int(is_tp))
            fp_values.append(int(is_fp))
            prediction_image_indices.append(image_index)

    if score_values:
        all_scores = np.asarray(score_values, dtype=np.float64)
        group_scores = np.unique(all_scores)[::-1]
        group_indices = np.searchsorted(-group_scores, -all_scores)
        prediction_image_array = np.asarray(prediction_image_indices, dtype=np.int32)
        tp_array = np.asarray(tp_values, dtype=np.float64)
        fp_array = np.asarray(fp_values, dtype=np.float64)
        group_count = len(group_scores)
    else:
        group_scores = np.empty((0,), dtype=np.float64)
        group_indices = np.empty((0,), dtype=np.int64)
        prediction_image_array = np.empty((0,), dtype=np.int32)
        tp_array = np.empty((0,), dtype=np.float64)
        fp_array = np.empty((0,), dtype=np.float64)
        group_count = 0

    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    ap_values: list[float] = []
    mean_latency_values: list[float] = []

    thresholds = np.asarray([1.0 - index / threshold_count for index in range(1, threshold_count + 1)])
    cutoff_groups = np.searchsorted(-group_scores, -thresholds, side="right") if group_count else np.zeros_like(thresholds, dtype=int)
    for sample in samples:
        image_multiplicity = np.bincount(sample, minlength=image_count).astype(np.float64)
        true_positives = float(np.dot(image_tp, image_multiplicity))
        false_positives = float(np.dot(image_fp, image_multiplicity))
        positive_count = float(np.dot(ground_truth_by_image, image_multiplicity))
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
        recall = true_positives / positive_count if positive_count else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
        mean_latency_values.append(float(np.mean(inference_ms[sample])))

        if group_count:
            prediction_weights = image_multiplicity[prediction_image_array]
            boot_tp_by_group = np.bincount(group_indices, weights=tp_array * prediction_weights, minlength=group_count)
            boot_fp_by_group = np.bincount(group_indices, weights=fp_array * prediction_weights, minlength=group_count)
            cumulative_tp = np.cumsum(boot_tp_by_group)
            cumulative_fp = np.cumsum(boot_fp_by_group)
            pr_points = []
            rc_points = []
            for cutoff in cutoff_groups:
                tp_at_threshold = float(cumulative_tp[cutoff - 1]) if cutoff else 0.0
                fp_at_threshold = float(cumulative_fp[cutoff - 1]) if cutoff else 0.0
                pr_points.append(tp_at_threshold / (tp_at_threshold + fp_at_threshold) if tp_at_threshold + fp_at_threshold else 0.0)
                rc_points.append(tp_at_threshold / positive_count if positive_count else 0.0)
            ap_values.append(_voc_ap(np.asarray(rc_points), np.asarray(pr_points)))
        else:
            ap_values.append(0.0)

    def interval(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        lower, upper = np.quantile(np.asarray(values), [0.025, 0.975])
        return {"lower": float(lower), "upper": float(upper)}

    return {
        "precision": interval(precisions),
        "recall": interval(recalls),
        "f1": interval(f1_values),
        "average_precision": interval(ap_values),
        "mean_inference_latency_ms": interval(mean_latency_values),
    }


def evaluate_detector(
    annotations: list[Annotation],
    predictions_by_image: dict[str, list[Detection]],
    inference_ms_by_image: dict[str, float],
    detector_metadata: dict[str, Any],
    iou_threshold: float,
    threshold_count: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, float]]]]:
    all_scores = [detection.score for detections in predictions_by_image.values() for detection in detections]
    score_min = min(all_scores) if all_scores else 0.0
    score_max = max(all_scores) if all_scores else 0.0
    score_range = score_max - score_min
    normalized_predictions = {
        image_id: [
            Detection(
                detection.bbox,
                (detection.score - score_min) / score_range if score_range > 0 else 1.0,
            )
            for detection in detections
        ]
        for image_id, detections in predictions_by_image.items()
    }

    results: dict[str, Any] = {
        "detector": detector_metadata,
        "images_evaluated": len(annotations),
        "raw_predictions": sum(len(detections) for detections in predictions_by_image.values()),
        "score_normalization": {"minimum": score_min, "maximum": score_max, "constant_scores_map_to": 1.0},
        "iou_threshold": iou_threshold,
        "operating_point": "all detections emitted by predeclared detector settings",
        "difficulty": {},
    }
    curves: dict[str, list[dict[str, float]]] = {}

    for difficulty in ("All", "Easy", "Medium", "Hard"):
        labels_by_image: list[list[tuple[float, bool, bool, bool]]] = []
        gt_counts: list[int] = []
        for annotation in annotations:
            labels, positive_count = classify_image_detections(
                annotation,
                normalized_predictions[annotation.image_id],
                difficulty,
                iou_threshold,
            )
            labels_by_image.append(labels)
            gt_counts.append(positive_count)

        tp = sum(int(row[1]) for rows in labels_by_image for row in rows)
        fp = sum(int(row[2]) for rows in labels_by_image for row in rows)
        ignored = sum(int(row[3]) for rows in labels_by_image for row in rows)
        ground_truth_count = sum(gt_counts)
        fn = ground_truth_count - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / ground_truth_count if ground_truth_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        flat_events = [
            row for rows in labels_by_image for row in rows if not row[3]
        ]
        flat_events.sort(key=lambda row: -row[0])
        if flat_events:
            event_scores = np.asarray([row[0] for row in flat_events], dtype=np.float64)
            group_scores = np.unique(event_scores)[::-1]
            group_indices = np.searchsorted(-group_scores, -event_scores)
            event_tp = np.asarray([int(row[1]) for row in flat_events], dtype=np.float64)
            event_fp = np.asarray([int(row[2]) for row in flat_events], dtype=np.float64)
            group_tp = np.bincount(group_indices, weights=event_tp, minlength=len(group_scores))
            group_fp = np.bincount(group_indices, weights=event_fp, minlength=len(group_scores))
        else:
            group_scores = np.empty((0,), dtype=np.float64)
            group_tp = np.empty((0,), dtype=np.float64)
            group_fp = np.empty((0,), dtype=np.float64)

        curve, average_precision = _curve_from_group_counts(
            group_scores, group_tp, group_fp, ground_truth_count, threshold_count
        )
        per_image_latency = np.asarray([inference_ms_by_image[annotation.image_id] for annotation in annotations], dtype=np.float64)
        confidence_intervals = _confidence_intervals(
            labels_by_image,
            np.asarray(gt_counts, dtype=np.float64),
            per_image_latency,
            bootstrap_seed,
            bootstrap_replicates,
            threshold_count,
        )
        results["difficulty"][difficulty] = {
            "ground_truth_faces": ground_truth_count,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "ignored_predictions": ignored,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": average_precision,
            "confidence_interval_95": confidence_intervals,
            "inference_latency_ms": {
                "mean": float(np.mean(per_image_latency)) if per_image_latency.size else 0.0,
                "median": float(np.median(per_image_latency)) if per_image_latency.size else 0.0,
                "p95": float(np.quantile(per_image_latency, 0.95)) if per_image_latency.size else 0.0,
            },
            "precision_recall_curve_points": len(curve),
        }
        curves[difficulty] = curve
    return results, curves


def load_predictions(path: str | Path, annotations: list[Annotation]) -> tuple[dict[str, list[Detection]], dict[str, float]]:
    predictions: dict[str, list[Detection]] = {}
    inference_ms: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            image_id = str(record["image_id"]).replace("\\", "/")
            if image_id in predictions:
                raise ValueError(f"Duplicate prediction entry for {image_id}")
            detection_rows = []
            for raw in record.get("detections", []):
                bbox = tuple(float(value) for value in raw["bbox"])
                score = float(raw["score"])
                if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox) or not math.isfinite(score):
                    raise ValueError(f"Invalid detection at {path}:{line_number}")
                if bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValueError(f"Non-positive detection box at {path}:{line_number}")
                detection_rows.append(Detection(bbox, score))
            elapsed = float(record["inference_ms"])
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError(f"Invalid inference latency at {path}:{line_number}")
            predictions[image_id] = detection_rows
            inference_ms[image_id] = elapsed

    expected = {annotation.image_id for annotation in annotations}
    actual = set(predictions)
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        unexpected = sorted(actual - expected)[:5]
        raise ValueError(f"Prediction/annotation image mismatch; missing={missing}, unexpected={unexpected}")
    return predictions, inference_ms


def write_pr_csv(path: str | Path, curves_by_detector: dict[str, dict[str, list[dict[str, float]]]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["detector", "difficulty", "score_threshold", "precision", "recall"])
        for detector, curves in curves_by_detector.items():
            for difficulty, points in curves.items():
                for point in points:
                    writer.writerow([detector, difficulty, point["score_threshold"], point["precision"], point["recall"]])
