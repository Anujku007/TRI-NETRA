import { createHash } from 'node:crypto';
import { createWriteStream, readFileSync } from 'node:fs';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { once } from 'node:events';
import { performance } from 'node:perf_hooks';
import { generateFaceDetections } from '../backend/services/faceEmbeddingService';

interface ExperimentConfig {
  widerFace: {
    imageRoot: string;
    predictionFiles: { humanjs: string };
    metadataFiles: { humanjs: string };
    humanjs: {
      warmupImages: number;
      scoreField: string;
      runtimeSettingsReference: string;
      maxDetectedPerImage: number;
      minimumDetectorConfidence: number;
    };
  };
}

interface Prediction {
  image_id: string;
  detections: Array<{ bbox: [number, number, number, number]; score: number }>;
  inference_ms: number;
}

const CONFIG_PATH = 'research-data/experiment-config.json';
const ANNOTATION_TEXT = 'research-data/widerface/wider_face_split/wider_face_val_bbx_gt.txt';
const MODEL_FILES = [
  'backend/models/blazeface.json',
  'backend/models/blazeface.bin',
  'backend/models/facemesh.json',
  'backend/models/facemesh.bin',
  'backend/models/faceres.json',
  'backend/models/faceres.bin',
  'backend/models/handtrack.json',
  'backend/models/handtrack.bin',
  'backend/models/handlandmark-lite.json',
  'backend/models/handlandmark-lite.bin',
];

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function loadImageIds(annotationPath: string): string[] {
  const lines = readFileSync(annotationPath, 'utf8').split(/\r?\n/);
  const imageIds: string[] = [];
  for (let index = 0; index < lines.length;) {
    const imageId = lines[index++].trim();
    if (!imageId) continue;
    const faceCount = Number(lines[index++]);
    if (!Number.isInteger(faceCount) || faceCount < 0) {
      throw new Error(`Invalid WIDER face count after ${imageId}`);
    }
    imageIds.push(imageId.toLowerCase().endsWith('.jpg') ? imageId : `${imageId}.jpg`);
    index += faceCount;
  }
  return imageIds;
}

async function hashFile(filePath: string): Promise<string> {
  return sha256(await readFile(filePath));
}

function toPrediction(imageId: string, detections: Awaited<ReturnType<typeof generateFaceDetections>>, inferenceMs: number): Prediction {
  return {
    image_id: imageId.replaceAll(path.sep, '/'),
    detections: detections.map((face) => ({
      bbox: face.box,
      score: face.confidence,
    })),
    inference_ms: inferenceMs,
  };
}

async function main() {
  const configBytes = await readFile(CONFIG_PATH);
  const config = JSON.parse(configBytes.toString('utf8')) as ExperimentConfig;
  const wider = config.widerFace;
  const imageIds = loadImageIds(ANNOTATION_TEXT);
  const imagePaths = imageIds.map((imageId) => path.join(wider.imageRoot, imageId));
  for (const imagePath of imagePaths) {
    if (!(await stat(imagePath).catch(() => null))) throw new Error(`Missing WIDER image: ${imagePath}`);
  }

  const outputPath = wider.predictionFiles.humanjs;
  const metadataPath = wider.metadataFiles.humanjs;
  await mkdir(path.dirname(outputPath), { recursive: true });
  const warmupCount = Math.min(wider.humanjs.warmupImages, imagePaths.length);
  console.log(`Human.js: ${imagePaths.length} WIDER validation images; warm-up=${warmupCount}`);

  for (const imagePath of imagePaths.slice(0, warmupCount)) {
    await generateFaceDetections(await readFile(imagePath));
  }

  const predictions: Prediction[] = [];
  let predictionCount = 0;
  let totalInferenceMs = 0;
  for (let index = 0; index < imagePaths.length; index++) {
    const bytes = await readFile(imagePaths[index]);
    const started = performance.now();
    const detections = await generateFaceDetections(bytes);
    const inferenceMs = performance.now() - started;
    totalInferenceMs += inferenceMs;
    const prediction = toPrediction(imageIds[index], detections, inferenceMs);
    predictionCount += prediction.detections.length;
    predictions.push(prediction);
    if ((index + 1) % 100 === 0 || index + 1 === imagePaths.length) {
      console.log(`Processed ${index + 1}/${imagePaths.length} images; detections=${predictionCount}`);
    }
  }
  const output = createWriteStream(outputPath, { encoding: 'utf8' });
  for (const prediction of predictions) output.write(`${JSON.stringify(prediction)}\n`);
  output.end();
  await once(output, 'finish');

  const packageJson = JSON.parse(await readFile('package.json', 'utf8')) as {
    dependencies: Record<string, string>;
  };
  const modelFileHashes = await Promise.all(MODEL_FILES.map(async (file) => ({
    path: file,
    sha256: await hashFile(file),
  })));
  const metadata = {
    detector: 'Human.js through the unchanged TRI-Netra backend face detector',
    dataset: 'WIDER FACE validation',
    annotationTextSha256: await hashFile(ANNOTATION_TEXT),
    experimentConfigSha256: sha256(configBytes),
    runtimeSettingsReference: wider.humanjs.runtimeSettingsReference,
    runtimeSettingsSha256: await hashFile(wider.humanjs.runtimeSettingsReference),
    scoreField: wider.humanjs.scoreField,
    maxDetectedPerImage: wider.humanjs.maxDetectedPerImage,
    minimumDetectorConfidence: wider.humanjs.minimumDetectorConfidence,
    warmupImages: warmupCount,
    imagesEvaluated: imagePaths.length,
    detections: predictionCount,
    totalInferenceMs,
    meanInferenceMs: totalInferenceMs / imagePaths.length,
    nodeVersion: process.version,
    packages: {
      human: packageJson.dependencies['@vladmandic/human'],
      tensorflowNode: packageJson.dependencies['@tensorflow/tfjs-node'],
    },
    modelFileHashes,
    predictionFileSha256: await hashFile(outputPath),
  };
  await writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);
  console.log(`Wrote ${outputPath} (${predictionCount} detections)`);
  console.log(`Wrote ${metadataPath}`);
}

main().catch((error: unknown) => {
  console.error('Human.js WIDER FACE evaluation failed:', error);
  process.exit(1);
});
