import { SampleScenario } from '@shared/types';
import { createSyntheticDemoScene } from '../utils/syntheticDemoScene';

export const SAMPLE_SCENARIOS: SampleScenario[] = [
  {
    id: 'scen-unauth-intruder',
    title: 'Restricted Vault Breach (Unauthorized Intruder)',
    category: 'UNAUTHORIZED',
    description: 'An unidentified person wearing a dark tactical jacket attempting unauthorized door bypass in the Server Vault.',
    imageUrl: createSyntheticDemoScene('Restricted vault breach', 'Server Vault', '#f45d62', 1, true),
    zone: 'Server Vault',
    expectedOutcome: 'Immediate Critical Threat Alert, unauthorized individual flagged, alarm triggered.',
  },
  {
    id: 'scen-auth-executive',
    title: 'Executive Corridor (Chief Security Officer Chen)',
    category: 'AUTHORIZED',
    description: 'CSO Sarah Chen entering the secure executive wing during regular hours.',
    imageUrl: createSyntheticDemoScene('Executive corridor', 'Executive Suite', '#42e0c0'),
    zone: 'Executive Suite',
    expectedOutcome: 'Status: Authorized. Clearance Level 4 verified. Zero alert.',
  },
  {
    id: 'scen-unauth-visitor',
    title: 'Loading Bay Loitering (Unidentified Subject)',
    category: 'UNAUTHORIZED',
    description: 'Unknown individual without security clearance badge standing near high-voltage distribution racks.',
    imageUrl: createSyntheticDemoScene('Loading bay review', 'Loading Bay', '#f2b84b'),
    zone: 'Loading Bay',
    expectedOutcome: 'High Threat Alert. Identity mismatch with all registered staff. Security dispatch suggested.',
  },
  {
    id: 'scen-auth-engineer',
    title: 'Server Vault Maintenance (David Okafor)',
    category: 'AUTHORIZED',
    description: 'Lead Security Architect David Okafor conducting scheduled telemetry review at terminal.',
    imageUrl: createSyntheticDemoScene('Vault maintenance', 'Server Vault', '#49b8ff'),
    zone: 'Server Vault',
    expectedOutcome: 'Status: Authorized. Clearance Level 3 verified. Access granted.',
  },
  {
    id: 'scen-mixed-group',
    title: 'Main Lobby Entry (Multiple Individuals)',
    category: 'MIXED',
    description: 'Group approaching the optical turnstiles containing authorized staff accompanied by an unverified guest.',
    imageUrl: createSyntheticDemoScene('Lobby group review', 'Main Entrance', '#b88cff', 2),
    zone: 'Main Entrance',
    expectedOutcome: 'Multi-face identification. Separate bounding boxes for verified vs unverified subjects.',
  },
];
