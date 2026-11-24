import unittest
from pathlib import Path

import numpy as np

from widerface_evaluation import (
    Annotation,
    Detection,
    _voc_ap,
    classify_image_detections,
    evaluate_detector,
    inclusive_iou,
    load_annotations,
)


ROOT = Path(__file__).resolve().parent.parent


class WiderFaceEvaluationTests(unittest.TestCase):
    def test_inclusive_iou_matches_official_boxoverlap_geometry(self):
        exact = inclusive_iou((0, 0, 9, 9), np.asarray([[0, 0, 9, 9]], dtype=np.float64))
        self.assertEqual(float(exact[0]), 1.0)
        one_pixel_intersection = inclusive_iou((9, 9, 9, 9), np.asarray([[0, 0, 9, 9]], dtype=np.float64))
        self.assertAlmostEqual(float(one_pixel_intersection[0]), 1.0 / 199.0)

    def test_difficulty_matching_ignores_non_kept_faces(self):
        annotation = Annotation(
            image_id="event/image.jpg",
            boxes=np.asarray([[0, 0, 9, 9], [20, 0, 9, 9]], dtype=np.float64),
            keep_indices={"Easy": {0}},
        )
        detections = [
            Detection((20, 0, 9, 9), 0.9),
            Detection((0, 0, 9, 9), 0.8),
            Detection((0, 0, 9, 9), 0.7),
            Detection((40, 40, 9, 9), 0.6),
        ]
        labels, count = classify_image_detections(annotation, detections, "Easy", 0.5)
        self.assertEqual(count, 1)
        self.assertEqual(labels, [
            (0.9, False, False, True),
            (0.8, True, False, False),
            (0.7, False, True, False),
            (0.6, False, True, False),
        ])

    def test_voc_precision_envelope_area(self):
        self.assertAlmostEqual(_voc_ap(np.asarray([0.5, 1.0]), np.asarray([1.0, 0.5])), 0.75)

    def test_shared_detector_metrics_and_image_bootstrap_are_deterministic(self):
        annotations = [
            Annotation(
                image_id="event/first.jpg",
                boxes=np.asarray([[0, 0, 9, 9]], dtype=np.float64),
                keep_indices={"Easy": {0}, "Medium": {0}, "Hard": {0}, "All": {0}},
            ),
            Annotation(
                image_id="event/second.jpg",
                boxes=np.asarray([[0, 0, 9, 9]], dtype=np.float64),
                keep_indices={"Easy": {0}, "Medium": {0}, "Hard": {0}, "All": {0}},
            ),
        ]
        predictions = {
            "event/first.jpg": [
                Detection((0, 0, 9, 9), 0.9),
                Detection((30, 30, 9, 9), 0.5),
            ],
            "event/second.jpg": [Detection((0, 0, 9, 9), 0.8)],
        }
        latency = {"event/first.jpg": 1.0, "event/second.jpg": 2.0}
        kwargs = {
            "annotations": annotations,
            "predictions_by_image": predictions,
            "inference_ms_by_image": latency,
            "detector_metadata": {"detector": "synthetic-test"},
            "iou_threshold": 0.5,
            "threshold_count": 10,
            "bootstrap_replicates": 40,
            "bootstrap_seed": 77,
        }
        first_report, first_curves = evaluate_detector(**kwargs)
        second_report, second_curves = evaluate_detector(**kwargs)
        easy = first_report["difficulty"]["Easy"]
        self.assertEqual((easy["true_positives"], easy["false_positives"], easy["false_negatives"]), (2, 1, 0))
        self.assertAlmostEqual(easy["precision"], 2 / 3)
        self.assertEqual(easy["recall"], 1.0)
        self.assertAlmostEqual(easy["f1"], 0.8)
        self.assertAlmostEqual(easy["average_precision"], 1.0)
        self.assertEqual(easy["inference_latency_ms"]["mean"], 1.5)
        self.assertEqual(first_report["difficulty"]["Easy"]["confidence_interval_95"], second_report["difficulty"]["Easy"]["confidence_interval_95"])
        self.assertEqual(first_curves, second_curves)

    def test_official_validation_subset_annotations_align(self):
        wider = ROOT / "research-data/widerface/wider_face_split"
        annotations = load_annotations(
            wider / "wider_face_val.mat",
            {
                "Easy": wider / "wider_easy_val.mat",
                "Medium": wider / "wider_medium_val.mat",
                "Hard": wider / "wider_hard_val.mat",
            },
        )
        self.assertEqual(len(annotations), 3226)
        self.assertTrue(all(set(("All", "Easy", "Medium", "Hard")).issubset(row.keep_indices) for row in annotations))
        self.assertTrue(all(all(index < len(row.boxes) for index in row.keep_indices["Hard"]) for row in annotations))


if __name__ == "__main__":
    unittest.main()
