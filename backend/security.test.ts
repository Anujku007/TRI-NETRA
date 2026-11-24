import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import { after, before, describe, it } from 'node:test';
import express from 'express';
import { SAMPLE_SCENARIOS } from '../frontend/src/data/sampleScenarios';
import { pruneIncidentRecords, readIncidentRetentionPolicy } from './incidentRetention';
import { fetchAllowedImage, isBlockedAddress } from './safeImageFetch';
import { installSensitiveApiAuthorization, type AdminSession } from './security';

const publicAddress = [{ address: '93.184.216.34', family: 4 }];

async function listen(server: Server): Promise<string> {
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  return `http://127.0.0.1:${address.port}`;
}

async function close(server: Server): Promise<void> {
  await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}

describe('sensitive API authorization', () => {
  let server: Server;
  let baseUrl: string;

  before(async () => {
    const app = express();
    const adminSession: AdminSession = {
      username: 'test-admin',
      role: 'SYSTEM_ADMINISTRATOR',
      loginTime: new Date().toISOString(),
      expiresAt: Date.now() + 60_000,
    };
    installSensitiveApiAuthorization(
      app,
      (request) => request.headers.authorization === 'Bearer admin-test' ? adminSession : null,
      (pairCode, token) => {
        if (pairCode !== 'PAIR-1') return null;
        if (token === 'cam-expired-token') return Date.now() - 1;
        return token === 'cam-test-token' ? Date.now() + 60_000 : null;
      },
    );
    app.use(express.json());
    app.all('/api/*', (_request, response) => response.json({ success: true }));
    server = createServer(app);
    baseUrl = await listen(server);
  });

  after(async () => {
    if (server?.listening) await close(server);
  });

  it('rejects unauthenticated reads and writes on every protected data API group', async () => {
    const paths = [
      '/api/personnel',
      '/api/tracking/presence',
      '/api/incidents/unauthorized',
      '/api/surveillance/analyze-frame',
    ];
    for (const path of paths) {
      const response = await fetch(`${baseUrl}${path}`, {
        method: 'POST',
        ...(path === '/api/personnel'
          ? { headers: { 'Content-Type': 'application/json' }, body: '{' }
          : {}),
      });
      assert.equal(response.status, 401, `${path} should require an admin session`);
    }
  });

  it('accepts an active admin session for protected APIs', async () => {
    const response = await fetch(`${baseUrl}/api/personnel`, {
      headers: { Authorization: 'Bearer admin-test' },
    });
    assert.equal(response.status, 200);
  });

  it('scopes camera and WebRTC access to a valid pair token and pair code', async () => {
    const noToken = await fetch(`${baseUrl}/api/phone-camera/latest-frame/PAIR-1`);
    const wrongPair = await fetch(`${baseUrl}/api/webrtc/signals/PAIR-2/phone`, {
      headers: { Authorization: 'Bearer cam-test-token' },
    });
    const expiredPair = await fetch(`${baseUrl}/api/webrtc/signals/PAIR-1/phone`, {
      headers: { Authorization: 'Bearer cam-expired-token' },
    });
    const validPair = await fetch(`${baseUrl}/api/phone-camera/latest-frame/PAIR-1`, {
      headers: { Authorization: 'Bearer cam-test-token' },
    });
    assert.equal(noToken.status, 401);
    assert.equal(wrongPair.status, 401);
    assert.equal(expiredPair.status, 401);
    assert.equal(validPair.status, 200);
  });
});

describe('safe remote image fetching', () => {
  it('blocks non-public IPv4 and IPv6 destinations', () => {
    const blocked = [
      '0.0.0.0', '10.1.2.3', '127.0.0.1', '169.254.169.254',
      '172.16.0.1', '192.168.1.10', '100.64.0.1', '224.0.0.1',
      '::', '::1', '::ffff:127.0.0.1', 'fc00::1', 'fe80::1', 'ff02::1',
      '2001:db8::1', '2002:7f00:1::1',
    ];
    for (const address of blocked) assert.equal(isBlockedAddress(address), true, address);
    assert.equal(isBlockedAddress('93.184.216.34'), false);
    assert.equal(isBlockedAddress('2606:4700:4700::1111'), false);
  });

  it('rejects hosts outside the allowlist before resolving or fetching', async () => {
    let resolverCalls = 0;
    let transportCalls = 0;
    await assert.rejects(fetchAllowedImage('https://other.example/image.jpg', {
      allowedHosts: ['allowed.example'],
      resolveHost: async () => { resolverCalls++; return publicAddress; },
      transport: async () => {
        transportCalls++;
        return { statusCode: 200, contentType: 'image/jpeg', body: new Uint8Array([1]) };
      },
    }), /allowlist/);
    assert.equal(resolverCalls, 0);
    assert.equal(transportCalls, 0);
  });

  it('rejects DNS answers that resolve to private addresses', async () => {
    let transportCalls = 0;
    await assert.rejects(fetchAllowedImage('https://allowed.example/image.jpg', {
      allowedHosts: ['allowed.example'],
      resolveHost: async () => [{ address: '10.0.0.5', family: 4 }],
      transport: async () => {
        transportCalls++;
        return { statusCode: 200, contentType: 'image/jpeg', body: new Uint8Array([1]) };
      },
    }), /blocked or non-public/);
    assert.equal(transportCalls, 0);
  });

  it('validates redirect destinations before issuing another request', async () => {
    let transportCalls = 0;
    await assert.rejects(fetchAllowedImage('https://allowed.example/start', {
      allowedHosts: ['allowed.example'],
      resolveHost: async () => publicAddress,
      transport: async () => {
        transportCalls++;
        return { statusCode: 302, location: 'http://127.0.0.1/internal', body: new Uint8Array() };
      },
    }), /allowlist|blocked IP/);
    assert.equal(transportCalls, 1);
  });

  it('enforces response byte limits and a timeout while resolving DNS', async () => {
    await assert.rejects(fetchAllowedImage('https://allowed.example/large', {
      allowedHosts: ['allowed.example'],
      maxBytes: 2,
      resolveHost: async () => publicAddress,
      transport: async () => ({
        statusCode: 200,
        contentType: 'image/jpeg',
        body: new Uint8Array([1, 2, 3]),
      }),
    }), /size limit/);

    await assert.rejects(fetchAllowedImage('https://allowed.example/slow-dns', {
      allowedHosts: ['allowed.example'],
      timeoutMs: 10,
      resolveHost: async () => new Promise(() => {}),
    }), /timed out/);

    await assert.rejects(fetchAllowedImage('https://allowed.example/slow-response', {
      allowedHosts: ['allowed.example'],
      timeoutMs: 10,
      resolveHost: async () => publicAddress,
      transport: async () => new Promise(() => {}),
    }), /timed out/);
  });

  it('returns bounded image data for an allowed public host', async () => {
    const image = await fetchAllowedImage('https://allowed.example/image.jpg', {
      allowedHosts: ['allowed.example'],
      resolveHost: async () => publicAddress,
      transport: async () => ({
        statusCode: 200,
        contentType: 'image/jpeg; charset=binary',
        body: new Uint8Array([1, 2, 3]),
      }),
    });
    assert.equal(image.mimeType, 'image/jpeg');
    assert.deepEqual([...image.body], [1, 2, 3]);
  });
});

describe('incident retention', () => {
  it('uses bounded retention settings and removes expired/excess records', () => {
    const policy = readIncidentRetentionPolicy({
      TRINETRA_INCIDENT_RETENTION_DAYS: '30',
      TRINETRA_INCIDENT_MAX_RECORDS: '2',
    } as NodeJS.ProcessEnv);
    const now = Date.parse('2026-01-31T00:00:00.000Z');
    const records = [
      { incident: { id: 'recent-a', timestamp: '2026-01-30T00:00:00.000Z' } },
      { incident: { id: 'expired', timestamp: '2025-01-01T00:00:00.000Z' } },
      { incident: { id: 'recent-b', timestamp: '2026-01-29T00:00:00.000Z' } },
      { incident: { id: 'recent-c', timestamp: '2026-01-28T00:00:00.000Z' } },
    ];
    const result = pruneIncidentRecords(records, now, policy);
    assert.deepEqual(result.retained.map(({ incident }) => incident.id), ['recent-a', 'recent-b']);
    assert.deepEqual(result.removed.map(({ incident }) => incident.id), ['expired', 'recent-c']);
  });
});

describe('presentation demo data', () => {
  it('uses local synthetic scenes without stock-person images', () => {
    assert.ok(SAMPLE_SCENARIOS.length > 0);
    for (const scenario of SAMPLE_SCENARIOS) {
      assert.match(scenario.imageUrl, /^data:image\/svg\+xml;charset=utf-8,/);
      assert.ok(!scenario.imageUrl.includes('http://') && !scenario.imageUrl.includes('https://'));
      assert.ok(scenario.imageUrl.includes('SYNTHETIC%20DEMO%20SCENE'));
    }
  });
});
