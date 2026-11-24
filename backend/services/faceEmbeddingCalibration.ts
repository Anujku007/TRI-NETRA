import { createHash } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import { extname, join, relative, resolve } from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import {
  FACE_IDENTITY_AUTHORIZATION_MARGIN,
  FACE_IDENTITY_AUTHORIZATION_THRESHOLD,
  FACE_IDENTITY_REVIEW_THRESHOLD,
  FaceIdentityMatcher,
  faceEmbeddingSimilarity,
} from '../../shared/utils/faceIdentityMatcher';
import { generateFaceEmbeddings } from './faceEmbeddingService';

import jpeg from 'jpeg-js';
import { PNG } from 'pngjs';

function decodeToRgb(buffer: Buffer): { data: Uint8Array; width: number; height: number } {
  const isPng = buffer[0] === 0x89 && buffer[1] === 0x50;
  const img: { data: Uint8Array | Buffer; width: number; height: number } = isPng
    ? PNG.sync.read(buffer)
    : jpeg.decode(buffer, { useTArray: true });
  const rgb = new Uint8Array(img.width * img.height * 3);
  for (let i = 0, j = 0; i < img.data.length; i += 4, j += 3) {
    rgb[j] = img.data[i];
    rgb[j + 1] = img.data[i + 1];
    rgb[j + 2] = img.data[i + 2];
  }
  return { data: rgb, width: img.width, height: img.height };
}

setWasmPaths(`${process.cwd()}/node_modules/@tensorflow/tfjs-backend-wasm/dist/`);
export const tfReady: Promise<void> = (async () => {
  await tf.setBackend('wasm');
  await tf.ready();
})();
tfReady.catch(() => undefined);

const LEGACY_CALIBRATION_THRESHOLDS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8];
export const FACE_IDENTITY_CALIBRATION_THRESHOLDS = [...new Set([
  FACE_IDENTITY_REVIEW_THRESHOLD,
  FACE_IDENTITY_AUTHORIZATION_THRESHOLD,
  ...LEGACY_CALIBRATION_THRESHOLDS,
])].sort((left, right) => left - right);

const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);
const MIN_IDENTITIES_FOR_ADEQUACY = 10;
const MIN_IMAGES_PER_IDENTITY_FOR_ADEQUACY = 5;
const MIN_PAIRS_PER_CLASS_FOR_ADEQUACY = 100;
const MAX_DUPLICATE_EXAMPLES = 200;

export interface CalibrationOptions {
  seed: number;
  calibrationFraction: number;
  maximumIdentities: number;
  minimumRawImagesPerIdentity: number;
  maximumImagesPerIdentity: number;
  identityBootstrapReplicates: number;
  additionalThresholds?: number[];
}

export interface CalibrationFaceImage {
  personId: string;
  filePath: string;
  split: 'calibration' | 'evaluation';
  embedding: number[];
  sha256: string;
  differenceHash: bigint;
}

export interface DuplicateAuditImage {
  personId: string;
  filePath: string;
  sha256: string;
  differenceHash: bigint;
}

export interface SkippedCalibrationImage {
  personId: string;
  filePath: string;
  reason: 'no-face-embedding' | 'multiple-faces' | 'decode-error' | 'exact-duplicate-same-identity';
  duplicateOf?: string;
}

export interface CalibrationThresholdResult {
  threshold: number;
  isRuntimeAuthorizationThreshold: boolean;
  samePersonPairCount: number;
  differentPersonPairCount: number;
  trueMatches: number;
  missedSamePersonMatches: number;
  falseDifferentPersonMatches: number;
  trueRejections: number;
  samePersonMatchRate: number | null;
  differentPersonFalseMatchRate: number | null;
  pairwisePrecision: number | null;
  pairwiseF1: number | null;
  identityClusterBootstrap95CI: {
    samePersonMatchRate: [number, number] | null;
    differentPersonFalseMatchRate: [number, number] | null;
    pairwisePrecision: [number, number] | null;
    pairwiseF1: [number, number] | null;
  };
}

export interface SimilarityDistribution {
  count: number;
  minimum: number | null;
  percentile25: number | null;
  median: number | null;
  percentile75: number | null;
  maximum: number | null;
}

export interface FaceIdentityCalibrationReport {
  schemaVersion: 2;
  datasetDirectory: string;
  selection: {
    seed: number;
    eligibleIdentityCount: number;
    selectedIdentityCount: number;
    requestedMaximumIdentities: number;
    minimumRawImagesPerIdentity: number;
    maximumImagesPerIdentity: number;
    sampledIdentities: string[];
  };
  split: {
    seed: number;
    calibrationFraction: number;
    calibrationIdentities: string[];
    evaluationIdentities: string[];
    identityDisjoint: boolean;
    crossIdentityExactByteGroupsKeptTogether: number;
  };
  usableImageCount: number;
  identityCount: number;
  usableImagesBySplit: { calibration: number; evaluation: number };
  usableIdentitiesBySplit: { calibration: number; evaluation: number };
  skippedImages: SkippedCalibrationImage[];
  duplicateAudit: {
    exactByteDuplicateGroupCount: number;
    exactSameIdentityDuplicatesRemoved: number;
    crossIdentityExactByteDuplicateGroupCount: number;
    coarseHashDistanceThreshold: number;
    coarseHashCandidatePairCount: number;
    coarseHashCrossIdentityCandidatePairCount: number;
    coarseHashCandidatesTruncated: boolean;
    coarseHashCandidates: Array<{
      firstFile: string;
      firstIdentity: string;
      secondFile: string;
      secondIdentity: string;
      hammingDistance: number;
    }>;
    policy: string;
  };
  runtimeScoring: {
    implementation: string;
    formula: string;
    authorizationThreshold: number;
    reviewThreshold: number;
    authorizationMargin: number;
    marginScope: string;
    calibrationUsesRuntimeFunction: true;
  };
  thresholds: number[];
  calibration: CalibrationSplitReport;
  evaluation: CalibrationSplitReport;
  adequacyRequirements: {
    minimumIdentities: number;
    minimumImagesPerIdentity: number;
    minimumPairsPerClass: number;
  };
}

export interface CalibrationSplitReport {
  usableImageCount: number;
  identityCount: number;
  imagesPerIdentity: Record<string, number>;
  samePersonScore: SimilarityDistribution;
  differentPersonScore: SimilarityDistribution;
  thresholdResults: CalibrationThresholdResult[];
  runtimeIdentification: {
    galleryImages: number;
    queryCount: number;
    rankOneAccuracy: number | null;
    acceptedCoverage: number | null;
    acceptedAccuracy: number | null;
    falseAcceptRatePerQuery: number | null;
    runtimeAuthorizationThreshold: number;
    runtimeAuthorizationMargin: number;
    identityClusterBootstrap95CI: {
      rankOneAccuracy: [number, number] | null;
      acceptedCoverage: [number, number] | null;
      acceptedAccuracy: [number, number] | null;
      falseAcceptRatePerQuery: [number, number] | null;
    };
  };
  sufficientForReliableCalibration: boolean;
}

type PendingImage = DuplicateAuditImage;

interface PairGroup {
  identityA: string;
  identityB: string;
  samePerson: boolean;
  scores: number[];
  hitsByThreshold: number[];
}

interface DuplicateAuditResult {
  retained: DuplicateAuditImage[];
  exactDuplicateGroups: number;
  exactSameIdentityDuplicatesRemoved: number;
  crossIdentityExactByteDuplicateGroupCount: number;
  coarseHashDistanceThreshold: number;
  coarseHashCandidatePairCount: number;
  coarseHashCrossIdentityCandidatePairCount: number;
  coarseHashCandidatesTruncated: boolean;
  coarseHashCandidates: FaceIdentityCalibrationReport['duplicateAudit']['coarseHashCandidates'];
  removed: Array<{ image: DuplicateAuditImage; duplicateOf: string }>;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function seedFor(seed: number, key: string): number {
  const hash = createHash('sha256').update(`${seed}:${key}`).digest();
  return hash.readUInt32LE(0);
}

function shuffled<T>(items: T[], random: () => number): T[] {
  const result = [...items];
  for (let index = result.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(random() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
}

export function createIdentityDisjointSplit(
  identities: string[],
  seed: number,
  calibrationFraction: number,
  exactByteLinkedIdentityGroups: string[][] = []
): { calibrationIdentities: string[]; evaluationIdentities: string[] } {
  if (new Set(identities).size !== identities.length) throw new Error('Identity list must not contain duplicates.');
  if (!Number.isFinite(calibrationFraction) || calibrationFraction <= 0 || calibrationFraction >= 1) {
    throw new RangeError('Calibration fraction must be between zero and one.');
  }
  if (identities.length < 2) throw new Error('At least two identities are required for a disjoint split.');

  const parent = new Map(identities.map((identity) => [identity, identity]));
  const findRoot = (identity: string): string => {
    const current = parent.get(identity);
    if (current === undefined) throw new Error(`Unknown identity in exact-byte group: ${identity}`);
    if (current === identity) return identity;
    const root = findRoot(current);
    parent.set(identity, root);
    return root;
  };
  for (const linkedGroup of exactByteLinkedIdentityGroups) {
    const members = [...new Set(linkedGroup)];
    if (members.some((identity) => !parent.has(identity))) {
      throw new Error('Exact-byte-linked identity group contains an identity outside the split set.');
    }
    for (const member of members.slice(1)) parent.set(findRoot(member), findRoot(members[0]));
  }

  const groupedIdentities = new Map<string, string[]>();
  for (const identity of identities) {
    const root = findRoot(identity);
    const group = groupedIdentities.get(root) ?? [];
    group.push(identity);
    groupedIdentities.set(root, group);
  }
  const orderedGroups = [...groupedIdentities.values()]
    .map((group) => group.sort((left, right) => left.localeCompare(right)))
    .sort((left, right) => left[0].localeCompare(right[0]));
  if (orderedGroups.length < 2) {
    throw new Error('Exact-byte duplicate links leave fewer than two independent identity groups to split.');
  }
  const randomizedGroups = shuffled(orderedGroups, mulberry32(seed >>> 0));
  const calibrationTarget = Math.min(
    identities.length - 1,
    Math.max(1, Math.floor(identities.length * calibrationFraction))
  );
  const calibrationGroups: string[][] = [];
  const evaluationGroups: string[][] = [];
  let calibrationCount = 0;
  for (let index = 0; index < randomizedGroups.length; index++) {
    const group = randomizedGroups[index];
    if (
      calibrationCount < calibrationTarget &&
      index < randomizedGroups.length - 1 &&
      calibrationCount + group.length < identities.length
    ) {
      calibrationGroups.push(group);
      calibrationCount += group.length;
    } else {
      evaluationGroups.push(group);
    }
  }
  if (calibrationGroups.length === 0) {
    const group = evaluationGroups.shift();
    if (!group || evaluationGroups.length === 0) throw new Error('Unable to create non-empty identity splits.');
    calibrationGroups.push(group);
  }
  const calibrationIdentities = calibrationGroups.flat();
  const evaluationIdentities = evaluationGroups.flat();
  return {
    calibrationIdentities: calibrationIdentities.sort((left, right) => left.localeCompare(right)),
    evaluationIdentities: evaluationIdentities.sort((left, right) => left.localeCompare(right)),
  };
}

function hashDistance(first: bigint, second: bigint): number {
  let bits = first ^ second;
  let distance = 0;
  while (bits !== 0n) {
    distance++;
    bits &= bits - 1n;
  }
  return distance;
}

export function auditDuplicateImages(
  images: DuplicateAuditImage[],
  coarseDistanceThreshold = 4
): DuplicateAuditResult {
  const exactGroups = new Map<string, DuplicateAuditImage[]>();
  for (const image of images) {
    const group = exactGroups.get(image.sha256) ?? [];
    group.push(image);
    exactGroups.set(image.sha256, group);
  }

  const removed = new Map<string, string>();
  let exactDuplicateGroups = 0;
  let exactSameIdentityDuplicatesRemoved = 0;
  let crossIdentityExactByteDuplicateGroupCount = 0;
  for (const group of exactGroups.values()) {
    if (group.length < 2) continue;
    exactDuplicateGroups++;
    const identities = new Set(group.map((image) => image.personId));
    if (identities.size !== 1) {
      crossIdentityExactByteDuplicateGroupCount++;
      continue;
    }
    const ordered = [...group].sort((left, right) => left.filePath.localeCompare(right.filePath));
    const keeper = ordered[0];
    for (const duplicate of ordered.slice(1)) {
      removed.set(duplicate.filePath, keeper.filePath);
      exactSameIdentityDuplicatesRemoved++;
    }
  }

  let coarseHashCandidatePairCount = 0;
  let coarseHashCrossIdentityCandidatePairCount = 0;
  const coarseHashCandidates: FaceIdentityCalibrationReport['duplicateAudit']['coarseHashCandidates'] = [];
  const orderedImages = [...images].sort((left, right) => left.filePath.localeCompare(right.filePath));
  for (let firstIndex = 0; firstIndex < orderedImages.length; firstIndex++) {
    const first = orderedImages[firstIndex];
    for (let secondIndex = firstIndex + 1; secondIndex < orderedImages.length; secondIndex++) {
      const second = orderedImages[secondIndex];
      const distance = hashDistance(first.differenceHash, second.differenceHash);
      if (distance > coarseDistanceThreshold) continue;
      coarseHashCandidatePairCount++;
      if (first.personId !== second.personId) coarseHashCrossIdentityCandidatePairCount++;
      if (coarseHashCandidates.length < MAX_DUPLICATE_EXAMPLES) {
        coarseHashCandidates.push({
          firstFile: first.filePath,
          firstIdentity: first.personId,
          secondFile: second.filePath,
          secondIdentity: second.personId,
          hammingDistance: distance,
        });
      }
    }
  }

  return {
    retained: orderedImages.filter((image) => !removed.has(image.filePath)),
    exactDuplicateGroups,
    exactSameIdentityDuplicatesRemoved,
    crossIdentityExactByteDuplicateGroupCount,
    coarseHashDistanceThreshold: coarseDistanceThreshold,
    coarseHashCandidatePairCount,
    coarseHashCrossIdentityCandidatePairCount,
    coarseHashCandidatesTruncated: coarseHashCandidatePairCount > coarseHashCandidates.length,
    coarseHashCandidates,
    removed: [...removed].map(([filePath, duplicateOf]) => ({
      image: orderedImages.find((image) => image.filePath === filePath)!,
      duplicateOf,
    })),
  };
}

function getDifferenceHash(imageBytes: Uint8Array): bigint {
  const decodedRgb = decodeToRgb(Buffer.from(imageBytes));
  const decoded = tf.tensor3d(decodedRgb.data, [decodedRgb.height, decodedRgb.width, 3], 'int32');
  const gray = decoded.mean(2);
  const expanded = tf.expandDims(gray, 0) as tf.Tensor3D;
  const resized = tf.image.resizeBilinear(expanded, [8, 9], false);

  try {
    const values = resized.dataSync();
    let hash = 0n;
    for (let row = 0; row < 8; row++) {
      for (let column = 0; column < 8; column++) {
        if (values[row * 9 + column] > values[row * 9 + column + 1]) {
          hash |= 1n << BigInt(row * 8 + column);
        }
      }
    }
    return hash;
  } finally {
    tf.dispose([decoded, gray, expanded, resized]);
  }
}

function getDistribution(values: number[]): SimilarityDistribution {
  if (values.length === 0) {
    return { count: 0, minimum: null, percentile25: null, median: null, percentile75: null, maximum: null };
  }
  const sorted = [...values].sort((left, right) => left - right);
  const percentile = (fraction: number) => {
    const index = (sorted.length - 1) * fraction;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    if (lower === upper) return sorted[lower];
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
  };
  return {
    count: sorted.length,
    minimum: sorted[0],
    percentile25: percentile(0.25),
    median: percentile(0.5),
    percentile75: percentile(0.75),
    maximum: sorted[sorted.length - 1],
  };
}

function makePairGroups(images: CalibrationFaceImage[], thresholds: number[]): PairGroup[] {
  const byIdentity = new Map<string, CalibrationFaceImage[]>();
  for (const image of images) {
    const group = byIdentity.get(image.personId) ?? [];
    group.push(image);
    byIdentity.set(image.personId, group);
  }
  const identities = [...byIdentity.keys()].sort((left, right) => left.localeCompare(right));
  const groups: PairGroup[] = [];
  const addGroup = (identityA: string, identityB: string, scores: number[]) => {
    if (scores.length === 0) return;
    groups.push({
      identityA,
      identityB,
      samePerson: identityA === identityB,
      scores,
      hitsByThreshold: thresholds.map((threshold) => scores.filter((score) => score >= threshold).length),
    });
  };

  for (let firstIdentityIndex = 0; firstIdentityIndex < identities.length; firstIdentityIndex++) {
    const firstIdentity = identities[firstIdentityIndex];
    const firstImages = byIdentity.get(firstIdentity)!;
    const sameScores: number[] = [];
    for (let firstImageIndex = 0; firstImageIndex < firstImages.length; firstImageIndex++) {
      for (let secondImageIndex = firstImageIndex + 1; secondImageIndex < firstImages.length; secondImageIndex++) {
        const score = faceEmbeddingSimilarity(firstImages[firstImageIndex].embedding, firstImages[secondImageIndex].embedding);
        if (score !== null) sameScores.push(score);
      }
    }
    addGroup(firstIdentity, firstIdentity, sameScores);

    for (let secondIdentityIndex = firstIdentityIndex + 1; secondIdentityIndex < identities.length; secondIdentityIndex++) {
      const secondIdentity = identities[secondIdentityIndex];
      const secondImages = byIdentity.get(secondIdentity)!;
      const differentScores: number[] = [];
      for (const firstImage of firstImages) {
        for (const secondImage of secondImages) {
          const score = faceEmbeddingSimilarity(firstImage.embedding, secondImage.embedding);
          if (score !== null) differentScores.push(score);
        }
      }
      addGroup(firstIdentity, secondIdentity, differentScores);
    }
  }
  return groups;
}

function percentileInterval(values: number[]): [number, number] | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const quantile = (fraction: number) => {
    const index = (sorted.length - 1) * fraction;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
  };
  return [quantile(0.025), quantile(0.975)];
}

function evaluatePairGroups(
  groups: PairGroup[],
  identities: string[],
  thresholds: number[],
  bootstrapReplicates: number,
  bootstrapSeed: number
): CalibrationThresholdResult[] {
  const classTotals = thresholds.map(() => ({ positive: 0, trueMatches: 0, negative: 0, falseMatches: 0 }));
  for (const group of groups) {
    for (let thresholdIndex = 0; thresholdIndex < thresholds.length; thresholdIndex++) {
      const counts = classTotals[thresholdIndex];
      if (group.samePerson) {
        counts.positive += group.scores.length;
        counts.trueMatches += group.hitsByThreshold[thresholdIndex];
      } else {
        counts.negative += group.scores.length;
        counts.falseMatches += group.hitsByThreshold[thresholdIndex];
      }
    }
  }

  const metricSamples = thresholds.map(() => ({ recall: [] as number[], falseMatchRate: [] as number[], precision: [] as number[], f1: [] as number[] }));
  if (bootstrapReplicates > 0 && identities.length > 1) {
    const random = mulberry32(bootstrapSeed >>> 0);
    for (let replicate = 0; replicate < bootstrapReplicates; replicate++) {
      const multiplicities = new Map<string, number>();
      for (let sample = 0; sample < identities.length; sample++) {
        const identity = identities[Math.floor(random() * identities.length)];
        multiplicities.set(identity, (multiplicities.get(identity) ?? 0) + 1);
      }
      const weighted = thresholds.map(() => ({ positive: 0, trueMatches: 0, negative: 0, falseMatches: 0 }));
      for (const group of groups) {
        const countA = multiplicities.get(group.identityA) ?? 0;
        if (countA === 0) continue;
        const weight = group.samePerson ? countA : countA * (multiplicities.get(group.identityB) ?? 0);
        if (weight === 0) continue;
        for (let thresholdIndex = 0; thresholdIndex < thresholds.length; thresholdIndex++) {
          const counts = weighted[thresholdIndex];
          if (group.samePerson) {
            counts.positive += group.scores.length * weight;
            counts.trueMatches += group.hitsByThreshold[thresholdIndex] * weight;
          } else {
            counts.negative += group.scores.length * weight;
            counts.falseMatches += group.hitsByThreshold[thresholdIndex] * weight;
          }
        }
      }
      for (let thresholdIndex = 0; thresholdIndex < thresholds.length; thresholdIndex++) {
        const counts = weighted[thresholdIndex];
        if (counts.positive === 0 || counts.negative === 0) continue;
        const recall = counts.trueMatches / counts.positive;
        const falseMatchRate = counts.falseMatches / counts.negative;
        const precisionDenominator = counts.trueMatches + counts.falseMatches;
        const precision = precisionDenominator ? counts.trueMatches / precisionDenominator : 0;
        const f1 = recall + precision ? (2 * recall * precision) / (recall + precision) : 0;
        metricSamples[thresholdIndex].recall.push(recall);
        metricSamples[thresholdIndex].falseMatchRate.push(falseMatchRate);
        metricSamples[thresholdIndex].precision.push(precision);
        metricSamples[thresholdIndex].f1.push(f1);
      }
    }
  }

  return thresholds.map((threshold, index) => {
    const counts = classTotals[index];
    const recall = counts.positive ? counts.trueMatches / counts.positive : null;
    const falseMatchRate = counts.negative ? counts.falseMatches / counts.negative : null;
    const precisionDenominator = counts.trueMatches + counts.falseMatches;
    const precision = precisionDenominator ? counts.trueMatches / precisionDenominator : null;
    const f1 = recall !== null && precision !== null && recall + precision
      ? (2 * recall * precision) / (recall + precision)
      : null;
    const samples = metricSamples[index];
    return {
      threshold,
      isRuntimeAuthorizationThreshold: threshold === FACE_IDENTITY_AUTHORIZATION_THRESHOLD,
      samePersonPairCount: counts.positive,
      differentPersonPairCount: counts.negative,
      trueMatches: counts.trueMatches,
      missedSamePersonMatches: counts.positive - counts.trueMatches,
      falseDifferentPersonMatches: counts.falseMatches,
      trueRejections: counts.negative - counts.falseMatches,
      samePersonMatchRate: recall,
      differentPersonFalseMatchRate: falseMatchRate,
      pairwisePrecision: precision,
      pairwiseF1: f1,
      identityClusterBootstrap95CI: {
        samePersonMatchRate: percentileInterval(samples.recall),
        differentPersonFalseMatchRate: percentileInterval(samples.falseMatchRate),
        pairwisePrecision: percentileInterval(samples.precision),
        pairwiseF1: percentileInterval(samples.f1),
      },
    };
  });
}

function evaluateRuntimeIdentification(
  images: CalibrationFaceImage[],
  bootstrapReplicates: number,
  bootstrapSeed: number
): CalibrationSplitReport['runtimeIdentification'] {
  const byIdentity = new Map<string, CalibrationFaceImage[]>();
  for (const image of images) {
    const identityImages = byIdentity.get(image.personId) ?? [];
    identityImages.push(image);
    byIdentity.set(image.personId, identityImages);
  }
  const identities = [...byIdentity.keys()].sort((left, right) => left.localeCompare(right));
  const matcher = new FaceIdentityMatcher();
  let galleryImages = 0;
  const orderedImagesByIdentity = new Map<string, CalibrationFaceImage[]>();
  const perIdentity = new Map<string, {
    queries: number;
    rankOneCorrect: number;
    accepted: number;
    acceptedCorrect: number;
    incorrectAccepted: number;
  }>();

  for (const identity of identities) {
    const identityImages = [...byIdentity.get(identity)!].sort((left, right) => left.filePath.localeCompare(right.filePath));
    if (identityImages.length === 0) continue;
    orderedImagesByIdentity.set(identity, identityImages);
    matcher.registerEmbedding(identity, identityImages[0].embedding);
    galleryImages++;
  }

  for (const identity of identities) {
    const identityImages = orderedImagesByIdentity.get(identity);
    if (!identityImages) continue;
    const counts = { queries: 0, rankOneCorrect: 0, accepted: 0, acceptedCorrect: 0, incorrectAccepted: 0 };
    for (const query of identityImages.slice(1)) {
      counts.queries++;
      const best = matcher.findBestMatch(query.embedding);
      const accepted = matcher.match(query.embedding);
      if (best?.personId === identity) counts.rankOneCorrect++;
      if (accepted) {
        counts.accepted++;
        if (accepted.personId === identity) counts.acceptedCorrect++;
        else counts.incorrectAccepted++;
      }
    }
    perIdentity.set(identity, counts);
  }

  const identitiesWithQueries = [...perIdentity.keys()].filter((identity) => perIdentity.get(identity)!.queries > 0);
  const aggregate = (multiplicities?: Map<string, number>) => {
    const totals = { queries: 0, rankOneCorrect: 0, accepted: 0, acceptedCorrect: 0, incorrectAccepted: 0 };
    for (const identity of identitiesWithQueries) {
      const weight = multiplicities ? (multiplicities.get(identity) ?? 0) : 1;
      const counts = perIdentity.get(identity)!;
      totals.queries += counts.queries * weight;
      totals.rankOneCorrect += counts.rankOneCorrect * weight;
      totals.accepted += counts.accepted * weight;
      totals.acceptedCorrect += counts.acceptedCorrect * weight;
      totals.incorrectAccepted += counts.incorrectAccepted * weight;
    }
    return totals;
  };
  const point = aggregate();
  const samples = {
    rankOneAccuracy: [] as number[],
    acceptedCoverage: [] as number[],
    acceptedAccuracy: [] as number[],
    falseAcceptRatePerQuery: [] as number[],
  };
  if (bootstrapReplicates > 0 && identitiesWithQueries.length > 1) {
    const random = mulberry32(bootstrapSeed >>> 0);
    for (let replicate = 0; replicate < bootstrapReplicates; replicate++) {
      const multiplicities = new Map<string, number>();
      for (let sample = 0; sample < identitiesWithQueries.length; sample++) {
        const identity = identitiesWithQueries[Math.floor(random() * identitiesWithQueries.length)];
        multiplicities.set(identity, (multiplicities.get(identity) ?? 0) + 1);
      }
      const totals = aggregate(multiplicities);
      if (!totals.queries) continue;
      samples.rankOneAccuracy.push(totals.rankOneCorrect / totals.queries);
      samples.acceptedCoverage.push(totals.accepted / totals.queries);
      samples.acceptedAccuracy.push(totals.accepted ? totals.acceptedCorrect / totals.accepted : 0);
      samples.falseAcceptRatePerQuery.push(totals.incorrectAccepted / totals.queries);
    }
  }
  return {
    galleryImages,
    queryCount: point.queries,
    rankOneAccuracy: point.queries ? point.rankOneCorrect / point.queries : null,
    acceptedCoverage: point.queries ? point.accepted / point.queries : null,
    acceptedAccuracy: point.accepted ? point.acceptedCorrect / point.accepted : null,
    falseAcceptRatePerQuery: point.queries ? point.incorrectAccepted / point.queries : null,
    runtimeAuthorizationThreshold: FACE_IDENTITY_AUTHORIZATION_THRESHOLD,
    runtimeAuthorizationMargin: FACE_IDENTITY_AUTHORIZATION_MARGIN,
    identityClusterBootstrap95CI: {
      rankOneAccuracy: percentileInterval(samples.rankOneAccuracy),
      acceptedCoverage: percentileInterval(samples.acceptedCoverage),
      acceptedAccuracy: percentileInterval(samples.acceptedAccuracy),
      falseAcceptRatePerQuery: percentileInterval(samples.falseAcceptRatePerQuery),
    },
  };
}

function summarizeSplit(
  images: CalibrationFaceImage[],
  thresholds: number[],
  adequacy: { minimumIdentities: number; minimumImagesPerIdentity: number; minimumPairsPerClass: number },
  bootstrapReplicates: number,
  seed: number,
  splitName: string
): CalibrationSplitReport {
  const counts = new Map<string, number>();
  for (const image of images) counts.set(image.personId, (counts.get(image.personId) ?? 0) + 1);
  const pairGroups = makePairGroups(images, thresholds);
  const sameScores = pairGroups.filter((group) => group.samePerson).flatMap((group) => group.scores);
  const differentScores = pairGroups.filter((group) => !group.samePerson).flatMap((group) => group.scores);
  const samePairCount = sameScores.length;
  const differentPairCount = differentScores.length;
  const identityNames = [...counts.keys()].sort((left, right) => left.localeCompare(right));
  const allIdentitiesHaveEnoughImages = identityNames.length > 0 && identityNames
    .every((identity) => (counts.get(identity) ?? 0) >= adequacy.minimumImagesPerIdentity);

  return {
    usableImageCount: images.length,
    identityCount: identityNames.length,
    imagesPerIdentity: Object.fromEntries([...counts.entries()].sort(([left], [right]) => left.localeCompare(right))),
    samePersonScore: getDistribution(sameScores),
    differentPersonScore: getDistribution(differentScores),
    thresholdResults: evaluatePairGroups(
      pairGroups,
      identityNames,
      thresholds,
      bootstrapReplicates,
      seedFor(seed, `identity-bootstrap:${splitName}`)
    ),
    runtimeIdentification: evaluateRuntimeIdentification(
      images,
      bootstrapReplicates,
      seedFor(seed, `runtime-identification-bootstrap:${splitName}`)
    ),
    sufficientForReliableCalibration:
      identityNames.length >= adequacy.minimumIdentities &&
      allIdentitiesHaveEnoughImages &&
      samePairCount >= adequacy.minimumPairsPerClass &&
      differentPairCount >= adequacy.minimumPairsPerClass,
  };
}

function defaultOptions(): CalibrationOptions {
  return {
    seed: 20261104,
    calibrationFraction: 0.6,
    maximumIdentities: 100,
    minimumRawImagesPerIdentity: 10,
    maximumImagesPerIdentity: 10,
    identityBootstrapReplicates: 1000,
  };
}

export function scoreCalibrationPair(first: ArrayLike<number>, second: ArrayLike<number>): number | null {
  return faceEmbeddingSimilarity(first, second);
}

export async function calibrateFaceIdentity(
  datasetDirectory: string,
  options: CalibrationOptions = defaultOptions()
): Promise<FaceIdentityCalibrationReport> {
  await tfReady;
  if (!Number.isInteger(options.seed) || options.seed < 0) throw new RangeError('Seed must be a non-negative integer.');
  if (!Number.isInteger(options.maximumIdentities) || options.maximumIdentities < 2) {
    throw new RangeError('Maximum identities must be at least two.');
  }
  if (!Number.isInteger(options.maximumImagesPerIdentity) || options.maximumImagesPerIdentity < 2) {
    throw new RangeError('Maximum images per identity must be at least two.');
  }
  if (!Number.isInteger(options.minimumRawImagesPerIdentity) || options.minimumRawImagesPerIdentity < 2) {
    throw new RangeError('Minimum raw images per identity must be at least two.');
  }
  if (!Number.isInteger(options.identityBootstrapReplicates) || options.identityBootstrapReplicates < 0) {
    throw new RangeError('Identity bootstrap replicate count must be a non-negative integer.');
  }

  const root = resolve(datasetDirectory);
  const personDirectories = (await readdir(root, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .sort((left, right) => left.name.localeCompare(right.name));
  const availableIdentities = [] as Array<{ personId: string; imageFiles: string[] }>;
  for (const personDirectory of personDirectories) {
    const imageFiles = (await readdir(join(root, personDirectory.name), { withFileTypes: true }))
      .filter((entry) => entry.isFile() && IMAGE_EXTENSIONS.has(extname(entry.name).toLowerCase()))
      .map((entry) => join(root, personDirectory.name, entry.name))
      .sort((left, right) => left.localeCompare(right));
    if (imageFiles.length >= options.minimumRawImagesPerIdentity) {
      availableIdentities.push({ personId: personDirectory.name, imageFiles });
    }
  }

  const selectedIdentities = shuffled(
    availableIdentities,
    mulberry32(options.seed >>> 0)
  ).slice(0, options.maximumIdentities).sort((left, right) => left.personId.localeCompare(right.personId));
  if (selectedIdentities.length < 2) {
    throw new Error(`Need at least two identities with ${options.minimumRawImagesPerIdentity} or more images.`);
  }
  const skippedImages: SkippedCalibrationImage[] = [];
  const pending: PendingImage[] = [];
  for (const { personId, imageFiles } of selectedIdentities) {
    const imageRandom = mulberry32(seedFor(options.seed, `image-sample:${personId}`));
    const selectedImageFiles = shuffled(imageFiles, imageRandom)
      .slice(0, options.maximumImagesPerIdentity)
      .sort((left, right) => left.localeCompare(right));
    for (const filePath of selectedImageFiles) {
      const relativePath = relative(root, filePath).split('\\').join('/');
      try {
        const imageBytes = await readFile(filePath);
        pending.push({
          personId,
          filePath: relativePath,
          sha256: createHash('sha256').update(imageBytes).digest('hex'),
          differenceHash: getDifferenceHash(imageBytes),
        });
      } catch {
        skippedImages.push({ personId, filePath: relativePath, reason: 'decode-error' });
      }
    }
  }

  const identitiesByExactHash = new Map<string, Set<string>>();
  for (const image of pending) {
    const identitiesForHash = identitiesByExactHash.get(image.sha256) ?? new Set<string>();
    identitiesForHash.add(image.personId);
    identitiesByExactHash.set(image.sha256, identitiesForHash);
  }
  const exactByteLinkedIdentityGroups = [...identitiesByExactHash.values()]
    .filter((group) => group.size > 1)
    .map((group) => [...group]);
  const identitySplit = createIdentityDisjointSplit(
    selectedIdentities.map((identity) => identity.personId),
    seedFor(options.seed, 'calibration-evaluation-split'),
    options.calibrationFraction,
    exactByteLinkedIdentityGroups
  );
  const splitByIdentity = new Map<string, 'calibration' | 'evaluation'>(
    identitySplit.calibrationIdentities.map((identity) => [identity, 'calibration'])
  );
  for (const identity of identitySplit.evaluationIdentities) splitByIdentity.set(identity, 'evaluation');

  const duplicateAudit = auditDuplicateImages(pending);
  for (const duplicate of duplicateAudit.removed) {
    skippedImages.push({
      personId: duplicate.image.personId,
      filePath: duplicate.image.filePath,
      reason: 'exact-duplicate-same-identity',
      duplicateOf: duplicate.duplicateOf,
    });
  }

  const usableImages: CalibrationFaceImage[] = [];
  for (const image of duplicateAudit.retained) {
    let detections;
    try {
      detections = await generateFaceEmbeddings(await readFile(join(root, image.filePath)));
    } catch {
      detections = [];
    }
    if (detections.length === 0) {
      skippedImages.push({ personId: image.personId, filePath: image.filePath, reason: 'no-face-embedding' });
    } else if (detections.length > 1) {
      skippedImages.push({ personId: image.personId, filePath: image.filePath, reason: 'multiple-faces' });
    } else {
      usableImages.push({
        ...image,
        split: splitByIdentity.get(image.personId)!,
        embedding: detections[0].embedding,
      });
    }
  }

  const thresholds = [...new Set([
    FACE_IDENTITY_REVIEW_THRESHOLD,
    FACE_IDENTITY_AUTHORIZATION_THRESHOLD,
    ...LEGACY_CALIBRATION_THRESHOLDS,
    ...(options.additionalThresholds ?? []),
  ])].sort((left, right) => left - right);
  if (thresholds.some((threshold) => !Number.isFinite(threshold) || threshold < 0 || threshold > 1)) {
    throw new RangeError('Calibration thresholds must be finite values between zero and one.');
  }

  const adequacyRequirements = {
    minimumIdentities: MIN_IDENTITIES_FOR_ADEQUACY,
    minimumImagesPerIdentity: MIN_IMAGES_PER_IDENTITY_FOR_ADEQUACY,
    minimumPairsPerClass: MIN_PAIRS_PER_CLASS_FOR_ADEQUACY,
  };
  const calibrationImages = usableImages.filter((image) => image.split === 'calibration');
  const evaluationImages = usableImages.filter((image) => image.split === 'evaluation');
  const crossSplitIdentities = new Set(calibrationImages.map((image) => image.personId));
  const identityDisjoint = evaluationImages.every((image) => !crossSplitIdentities.has(image.personId));
  if (!identityDisjoint) throw new Error('Calibration/evaluation identity leakage detected.');
  const exactByteGroupsKeptTogether = exactByteLinkedIdentityGroups.every((group) => {
    const split = splitByIdentity.get(group[0]);
    return group.every((identity) => splitByIdentity.get(identity) === split);
  });
  if (!exactByteGroupsKeptTogether) throw new Error('Exact-byte duplicate leakage across calibration/evaluation splits detected.');

  return {
    schemaVersion: 2,
    datasetDirectory: relative(process.cwd(), root).split('\\').join('/') || '.',
    selection: {
      seed: options.seed,
      eligibleIdentityCount: availableIdentities.length,
      selectedIdentityCount: selectedIdentities.length,
      requestedMaximumIdentities: options.maximumIdentities,
      minimumRawImagesPerIdentity: options.minimumRawImagesPerIdentity,
      maximumImagesPerIdentity: options.maximumImagesPerIdentity,
      sampledIdentities: selectedIdentities.map((identity) => identity.personId),
    },
    split: {
      seed: seedFor(options.seed, 'calibration-evaluation-split'),
      calibrationFraction: options.calibrationFraction,
      calibrationIdentities: identitySplit.calibrationIdentities,
      evaluationIdentities: identitySplit.evaluationIdentities,
      identityDisjoint,
      crossIdentityExactByteGroupsKeptTogether: exactByteLinkedIdentityGroups.length,
    },
    usableImageCount: usableImages.length,
    identityCount: new Set(usableImages.map((image) => image.personId)).size,
    usableImagesBySplit: { calibration: calibrationImages.length, evaluation: evaluationImages.length },
    usableIdentitiesBySplit: {
      calibration: new Set(calibrationImages.map((image) => image.personId)).size,
      evaluation: new Set(evaluationImages.map((image) => image.personId)).size,
    },
    skippedImages,
    duplicateAudit: {
      exactByteDuplicateGroupCount: duplicateAudit.exactDuplicateGroups,
      exactSameIdentityDuplicatesRemoved: duplicateAudit.exactSameIdentityDuplicatesRemoved,
      crossIdentityExactByteDuplicateGroupCount: duplicateAudit.crossIdentityExactByteDuplicateGroupCount,
      coarseHashDistanceThreshold: duplicateAudit.coarseHashDistanceThreshold,
      coarseHashCandidatePairCount: duplicateAudit.coarseHashCandidatePairCount,
      coarseHashCrossIdentityCandidatePairCount: duplicateAudit.coarseHashCrossIdentityCandidatePairCount,
      coarseHashCandidatesTruncated: duplicateAudit.coarseHashCandidatesTruncated,
      coarseHashCandidates: duplicateAudit.coarseHashCandidates,
      policy: 'Coarse difference hashes only flag review candidates. They never discard images. Exact-byte duplicates are removed only within one identity; cross-identity exact-byte duplicates are retained and reported.',
    },
    runtimeScoring: {
      implementation: 'shared/utils/faceIdentityMatcher.ts::faceEmbeddingSimilarity',
      formula: 'd = round(100 * 25 * sum((a[i]-b[i])^2)) / 100; s = round(100 * clamp((1 - sqrt(d)/100 - 0.2)/(0.8-0.2), 0, 1)) / 100',
      authorizationThreshold: FACE_IDENTITY_AUTHORIZATION_THRESHOLD,
      reviewThreshold: FACE_IDENTITY_REVIEW_THRESHOLD,
      authorizationMargin: FACE_IDENTITY_AUTHORIZATION_MARGIN,
      marginScope: 'Applied in the leave-one-image-out gallery identification evaluation through the runtime FaceIdentityMatcher; pairwise verification metrics use the runtime similarity threshold.',
      calibrationUsesRuntimeFunction: true,
    },
    thresholds,
    calibration: summarizeSplit(
      calibrationImages,
      thresholds,
      adequacyRequirements,
      options.identityBootstrapReplicates,
      options.seed,
      'calibration'
    ),
    evaluation: summarizeSplit(
      evaluationImages,
      thresholds,
      adequacyRequirements,
      options.identityBootstrapReplicates,
      options.seed,
      'evaluation'
    ),
    adequacyRequirements,
  };
}

export function formatFaceIdentityCalibrationReport(report: FaceIdentityCalibrationReport): string {
  return `${JSON.stringify(report, null, 2)}\n`;
}
