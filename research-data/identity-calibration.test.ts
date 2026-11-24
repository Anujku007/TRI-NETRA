import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FACE_IDENTITY_AUTHORIZATION_THRESHOLD,
  faceEmbeddingSimilarity,
} from '../shared/utils/faceIdentityMatcher';
import {
  auditDuplicateImages,
  createIdentityDisjointSplit,
  scoreCalibrationPair,
} from '../backend/services/faceEmbeddingCalibration';

test('calibration delegates score calculation to the runtime identity scorer', () => {
  const first = [0, 0.25, -0.75, 1.1];
  const second = [0.1, 0.2, -0.7, -0.9];
  assert.equal(scoreCalibrationPair(first, second), faceEmbeddingSimilarity(first, second));
  assert.equal(scoreCalibrationPair(first, [0, 1]), null);
  assert.equal(FACE_IDENTITY_AUTHORIZATION_THRESHOLD, 0.42);
  assert.equal(scoreCalibrationPair([1], [6.6]), 0.87);
});

test('exact-byte-linked identities stay together in the disjoint split', () => {
  const split = createIdentityDisjointSplit(
    ['p1', 'p2', 'p3', 'p4', 'p5', 'p6'],
    44,
    0.5,
    [['p1', 'p4'], ['p2', 'p3']]
  );
  const splitFor = (identity: string) => split.calibrationIdentities.includes(identity) ? 'calibration' : 'evaluation';
  assert.equal(splitFor('p1'), splitFor('p4'));
  assert.equal(splitFor('p2'), splitFor('p3'));
  assert.ok(split.calibrationIdentities.length > 0);
  assert.ok(split.evaluationIdentities.length > 0);
});

test('identity split is deterministic, complete, and disjoint', () => {
  const identities = ['p6', 'p1', 'p2', 'p5', 'p4', 'p3'];
  const first = createIdentityDisjointSplit(identities, 12345, 0.5);
  const second = createIdentityDisjointSplit(identities, 12345, 0.5);
  assert.deepEqual(first, second);
  assert.equal(first.calibrationIdentities.length, 3);
  assert.equal(first.evaluationIdentities.length, 3);
  assert.deepEqual(
    [...first.calibrationIdentities, ...first.evaluationIdentities].sort(),
    [...identities].sort()
  );
  assert.equal(first.calibrationIdentities.some((identity) => first.evaluationIdentities.includes(identity)), false);
});

test('coarse hash collisions are reported and retained across identities', () => {
  const report = auditDuplicateImages([
    { personId: 'alice', filePath: 'alice/one.jpg', sha256: 'a', differenceHash: 0n },
    { personId: 'bob', filePath: 'bob/one.jpg', sha256: 'b', differenceHash: 0n },
  ]);
  assert.equal(report.coarseHashCandidatePairCount, 1);
  assert.equal(report.coarseHashCrossIdentityCandidatePairCount, 1);
  assert.deepEqual(report.retained.map((image) => image.personId), ['alice', 'bob']);
  assert.equal(report.exactSameIdentityDuplicatesRemoved, 0);
});

test('exact duplicates are removed only within one identity', () => {
  const report = auditDuplicateImages([
    { personId: 'alice', filePath: 'alice/a.jpg', sha256: 'same-bytes', differenceHash: 0n },
    { personId: 'alice', filePath: 'alice/b.jpg', sha256: 'same-bytes', differenceHash: 0n },
    { personId: 'alice', filePath: 'alice/cross.jpg', sha256: 'cross-id-bytes', differenceHash: 9n },
    { personId: 'bob', filePath: 'bob/cross.jpg', sha256: 'cross-id-bytes', differenceHash: 9n },
  ]);
  assert.equal(report.exactSameIdentityDuplicatesRemoved, 1);
  assert.equal(report.crossIdentityExactByteDuplicateGroupCount, 1);
  assert.equal(report.retained.length, 3);
  assert.deepEqual(report.removed.map((entry) => entry.image.filePath), ['alice/b.jpg']);
});
