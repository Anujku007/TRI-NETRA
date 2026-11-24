"""Generate OpenCV Haar predictions for the shared WIDER FACE evaluator."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from widerface_evaluation import load_annotations


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "research-data/experiment-config.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer(cascade: cv2.CascadeClassifier, image_bytes: bytes) -> tuple[list[list[int]], list[float]]:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode a WIDER validation image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, _reject_levels, level_weights = cascade.detectMultiScale3(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20),
        outputRejectLevels=True,
    )
    return (
        [list(map(int, box)) for box in boxes],
        [float(weight) for weight in np.asarray(level_weights).reshape(-1)],
    )


def main() -> None:
    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    wider = config["widerFace"]
    annotation_path = ROOT / wider["annotationMat"]
    difficulty_paths = {name: ROOT / value for name, value in wider["difficultyAnnotations"].items()}
    annotations = load_annotations(annotation_path, difficulty_paths)
    if len(annotations) != wider["expectedImageCount"]:
        raise ValueError(f"Expected {wider['expectedImageCount']} WIDER images, got {len(annotations)}")

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV Haar cascade: {cascade_path}")

    image_root = ROOT / wider["imageRoot"]
    image_paths = [(annotation.image_id, image_root / annotation.image_id) for annotation in annotations]
    for image_id, image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing WIDER image {image_id}: {image_path}")

    warmup_count = min(int(wider["haar"]["warmupImages"]), len(image_paths))
    print(f"OpenCV Haar: {len(image_paths)} WIDER validation images; warm-up={warmup_count}")
    for _, image_path in image_paths[:warmup_count]:
        infer(cascade, image_path.read_bytes())

    output_path = ROOT / wider["predictionFiles"]["haar"]
    metadata_path = ROOT / wider["metadataFiles"]["haar"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    prediction_count = 0
    for index, (image_id, image_path) in enumerate(image_paths, start=1):
        image_bytes = image_path.read_bytes()
        started = time.perf_counter_ns()
        boxes, scores = infer(cascade, image_bytes)
        inference_ms = (time.perf_counter_ns() - started) / 1_000_000
        if len(boxes) != len(scores):
            raise RuntimeError(f"Haar returned {len(boxes)} boxes and {len(scores)} scores for {image_id}")
        detections = [
            {"bbox": box, "score": score}
            for box, score in zip(boxes, scores)
        ]
        prediction_count += len(detections)
        predictions.append({
            "image_id": image_id,
            "detections": detections,
            "inference_ms": inference_ms,
        })
        if index % 100 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)} images; detections={prediction_count}", flush=True)

    with output_path.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, separators=(",", ":")) + "\n")

    metadata = {
        "detector": "OpenCV Haar Cascade (haarcascade_frontalface_default)",
        "dataset": "WIDER FACE validation",
        "annotationMatSha256": sha256_file(annotation_path),
        "experimentConfigSha256": hashlib.sha256(config_bytes).hexdigest(),
        "cascadePath": str(cascade_path),
        "cascadeSha256": sha256_file(cascade_path),
        "warmupImages": warmup_count,
        "imagesEvaluated": len(annotations),
        "detections": prediction_count,
        "settings": wider["haar"],
        "scoreField": "OpenCV detectMultiScale3 levelWeight; raw ranking score, not a calibrated probability",
        "pythonVersion": platform.python_version(),
        "opencvVersion": cv2.__version__,
        "numpyVersion": np.__version__,
        "predictionFileSha256": sha256_file(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path.relative_to(ROOT)} ({prediction_count} detections)")
    print(f"Wrote {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
