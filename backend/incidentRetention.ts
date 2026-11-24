export interface RetentionPolicy {
  retentionDays: number;
  maxRecords: number;
}

export interface RetainedIncidentRecord {
  incident: { id: string; timestamp?: string; [key: string]: unknown };
}

export function readIncidentRetentionPolicy(env: NodeJS.ProcessEnv = process.env): RetentionPolicy {
  const readBoundedInteger = (value: string | undefined, fallback: number, maximum: number) => {
    const parsed = Number(value);
    return Number.isSafeInteger(parsed) && parsed > 0 ? Math.min(parsed, maximum) : fallback;
  };
  return {
    retentionDays: readBoundedInteger(env.TRINETRA_INCIDENT_RETENTION_DAYS, 90, 36_500),
    maxRecords: readBoundedInteger(env.TRINETRA_INCIDENT_MAX_RECORDS, 1_000, 100_000),
  };
}

export function pruneIncidentRecords<TRecord extends RetainedIncidentRecord>(
  records: readonly TRecord[],
  now: number,
  policy: RetentionPolicy,
): { retained: TRecord[]; removed: TRecord[] } {
  const cutoff = now - policy.retentionDays * 24 * 60 * 60 * 1_000;
  const fresh = records.filter((record) => {
    const timestamp = Date.parse(record.incident.timestamp || '');
    return Number.isFinite(timestamp) && timestamp >= cutoff;
  }).sort((left, right) =>
    Date.parse(right.incident.timestamp || '') - Date.parse(left.incident.timestamp || '')
  );
  const retained = fresh.slice(0, policy.maxRecords);
  const kept = new Set(retained);
  return { retained, removed: records.filter((record) => !kept.has(record)) };
}
