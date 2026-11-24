import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import {
  calibrateFaceIdentity,
  formatFaceIdentityCalibrationReport,
} from '../services/faceEmbeddingCalibration';

async function main() {
  const configPath = 'research-data/experiment-config.json';
  const config = JSON.parse(await readFile(configPath, 'utf8'));
  const experiment = config.identityCalibration;
  const datasetDirectory = process.argv[2] || experiment.datasetDirectory;
  if (!datasetDirectory) throw new Error(`Missing datasetDirectory in ${configPath}`);

  const options = {
    seed: experiment.seed,
    calibrationFraction: experiment.calibrationFraction,
    maximumIdentities: experiment.maximumIdentities,
    minimumRawImagesPerIdentity: experiment.minimumRawImagesPerIdentity,
    maximumImagesPerIdentity: experiment.maximumImagesPerIdentity,
    identityBootstrapReplicates: experiment.identityBootstrapReplicates,
    additionalThresholds: experiment.additionalThresholds || [],
  };

  console.log(`Starting identity-disjoint face calibration: ${datasetDirectory}`);
  console.log(`Seed: ${options.seed}; identity sample cap: ${options.maximumIdentities}`);

  const report = await calibrateFaceIdentity(datasetDirectory, options);
  const formattedReport = formatFaceIdentityCalibrationReport(report);
  const outputPaths = experiment.outputFiles || ['research-data/results/identity_calibration.json'];
  for (const outputPath of outputPaths) {
    await mkdir(dirname(outputPath), { recursive: true });
    await writeFile(outputPath, formattedReport, 'utf8');
  }

  console.log('\nCalibration completed.');
  console.log(`Usable images: ${report.usableImageCount}`);
  console.log(`Identities: ${report.identityCount}`);
  console.log(`Identity disjoint: ${report.split.identityDisjoint}`);
  console.log(`Calibration identities: ${report.calibration.identityCount}`);
  console.log(`Evaluation identities: ${report.evaluation.identityCount}`);
  console.log(`Calibration adequate: ${report.calibration.sufficientForReliableCalibration}`);
  console.log(`Evaluation adequate: ${report.evaluation.sufficientForReliableCalibration}`);
  console.log(`Reports saved: ${outputPaths.join(', ')}`);

  for (const [splitName, split] of [['calibration', report.calibration], ['evaluation', report.evaluation]] as const) {
    console.log(`\n${splitName} thresholds:`);
    for (const result of split.thresholdResults) {
      console.log({
        threshold: result.threshold,
        runtimeThreshold: result.isRuntimeAuthorizationThreshold,
        samePersonMatchRate: result.samePersonMatchRate,
        differentPersonFalseMatchRate: result.differentPersonFalseMatchRate,
        samePersonPairCount: result.samePersonPairCount,
        differentPersonPairCount: result.differentPersonPairCount,
      });
    }
  }
}

main().catch((error) => {
  console.error('Calibration failed:', error);
  process.exit(1);
});
