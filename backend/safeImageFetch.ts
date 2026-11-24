import { lookup } from 'node:dns/promises';
import { request as httpRequest, type RequestOptions as HttpRequestOptions } from 'node:http';
import { request as httpsRequest, type RequestOptions as HttpsRequestOptions } from 'node:https';
import { isIP } from 'node:net';

export interface ResolvedAddress {
  address: string;
  family: number;
}

export interface SafeImageResponse {
  statusCode: number;
  contentType?: string;
  contentLength?: string;
  location?: string;
  body: Uint8Array;
}

export type SafeImageResolver = (hostname: string) => Promise<ResolvedAddress[]>;
export type SafeImageTransport = (
  url: URL,
  addresses: readonly ResolvedAddress[],
  signal: AbortSignal,
  maxBytes: number,
) => Promise<SafeImageResponse>;

export interface SafeImageFetchOptions {
  allowedHosts: ReadonlySet<string> | readonly string[];
  maxBytes?: number;
  timeoutMs?: number;
  maxRedirects?: number;
  resolveHost?: SafeImageResolver;
  transport?: SafeImageTransport;
}

const DEFAULT_MAX_BYTES = 8 * 1024 * 1024;
const DEFAULT_TIMEOUT_MS = 4_000;
const DEFAULT_MAX_REDIRECTS = 2;
const ACCEPTED_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);

function normalizeHost(hostname: string): string {
  return hostname.toLowerCase().replace(/^\[|\]$/g, '').replace(/\.$/, '');
}

export function parseAllowedImageHosts(value: string | undefined): Set<string> {
  const hosts = new Set<string>();
  for (const entry of (value || '').split(',')) {
    const host = normalizeHost(entry.trim());
    if (!host || host.includes('*') || host.includes('/') || host.includes(':') || /\s/.test(host)) continue;
    if (/^[a-z0-9.-]+$/.test(host) && !host.startsWith('.') && !host.endsWith('.')) hosts.add(host);
  }
  return hosts;
}

function ipv4Number(address: string): number | null {
  const octets = address.split('.').map(Number);
  if (octets.length !== 4 || octets.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) return null;
  return (((octets[0] * 256 + octets[1]) * 256 + octets[2]) * 256 + octets[3]) >>> 0;
}

function ipv4InRange(address: number, network: number, prefixLength: number): boolean {
  const mask = prefixLength === 0 ? 0 : (0xffffffff << (32 - prefixLength)) >>> 0;
  return (address & mask) === (network & mask);
}

function isBlockedIPv4(address: string): boolean {
  const numeric = ipv4Number(address);
  if (numeric === null) return true;
  const blockedRanges: Array<[string, number]> = [
    ['0.0.0.0', 8], ['10.0.0.0', 8], ['100.64.0.0', 10], ['127.0.0.0', 8],
    ['169.254.0.0', 16], ['172.16.0.0', 12], ['192.0.0.0', 24],
    ['192.0.2.0', 24], ['192.88.99.0', 24], ['192.168.0.0', 16],
    ['198.18.0.0', 15], ['198.51.100.0', 24], ['203.0.113.0', 24],
    ['224.0.0.0', 4], ['240.0.0.0', 4],
  ];
  return blockedRanges.some(([network, prefix]) => {
    const parsedNetwork = ipv4Number(network);
    return parsedNetwork !== null && ipv4InRange(numeric, parsedNetwork, prefix);
  });
}

function ipv6Number(address: string): bigint | null {
  if (address.includes('%')) return null;
  let value = address.toLowerCase();
  const lastColon = value.lastIndexOf(':');
  const possibleIPv4 = value.slice(lastColon + 1);
  if (possibleIPv4.includes('.')) {
    const ipv4 = ipv4Number(possibleIPv4);
    if (ipv4 === null) return null;
    const high = ((ipv4 >>> 16) & 0xffff).toString(16);
    const low = (ipv4 & 0xffff).toString(16);
    value = `${value.slice(0, lastColon + 1)}${high}:${low}`;
  }

  const sections = value.split('::');
  if (sections.length > 2) return null;
  const left = sections[0] ? sections[0].split(':') : [];
  const right = sections.length === 2 && sections[1] ? sections[1].split(':') : [];
  if (left.concat(right).some((part) => !/^[a-f0-9]{1,4}$/.test(part))) return null;
  const missing = 8 - left.length - right.length;
  if ((sections.length === 1 && missing !== 0) || (sections.length === 2 && missing < 1)) return null;
  const groups = [...left, ...Array(Math.max(0, missing)).fill('0'), ...right];
  if (groups.length !== 8) return null;
  return groups.reduce((acc, part) => (acc << 16n) | BigInt(`0x${part}`), 0n);
}

function ipv6InRange(address: bigint, network: bigint, prefixLength: number): boolean {
  const mask = ((1n << 128n) - 1n) ^ ((1n << BigInt(128 - prefixLength)) - 1n);
  return (address & mask) === (network & mask);
}

function isBlockedIPv6(address: string): boolean {
  const numeric = ipv6Number(address);
  if (numeric === null) return true;

  // Accept only global-unicast space. This rejects loopback, unspecified,
  // unique-local, link-local, multicast, IPv4-compatible and reserved ranges.
  const globalUnicast = ipv6InRange(numeric, 0x20000000000000000000000000000000n, 3);
  if (!globalUnicast) return true;

  // Documentation and transition space is not globally routable in practice.
  if (ipv6InRange(numeric, 0x20010db8000000000000000000000000n, 32)) return true;
  if (ipv6InRange(numeric, 0x20010000000000000000000000000000n, 23)) return true;
  // 6to4 embeds an IPv4 destination and can otherwise bypass IPv4 range checks.
  if (ipv6InRange(numeric, 0x20020000000000000000000000000000n, 16)) return true;

  // IPv4-mapped IPv6 must be checked against the embedded IPv4 destination.
  if (ipv6InRange(numeric, 0x00000000000000000000ffff00000000n, 96)) {
    const embedded = Number(numeric & 0xffffffffn);
    const embeddedAddress = [
      (embedded >>> 24) & 255,
      (embedded >>> 16) & 255,
      (embedded >>> 8) & 255,
      embedded & 255,
    ].join('.');
    return isBlockedIPv4(embeddedAddress);
  }
  return false;
}

export function isBlockedAddress(address: string): boolean {
  const family = isIP(address);
  if (family === 4) return isBlockedIPv4(address);
  if (family === 6) return isBlockedIPv6(address);
  return true;
}

async function resolvePublicAddresses(hostname: string): Promise<ResolvedAddress[]> {
  const cleanHostname = normalizeHost(hostname);
  const family = isIP(cleanHostname);
  const addresses = family
    ? [{ address: cleanHostname, family }]
    : await lookup(cleanHostname, { all: true, verbatim: true });
  if (addresses.length === 0 || addresses.some(({ address }) => isBlockedAddress(address))) {
    throw new Error('Image host resolves to a blocked or non-public IP address.');
  }
  return addresses;
}

function normalizeAllowedHosts(allowedHosts: SafeImageFetchOptions['allowedHosts']): Set<string> {
  return new Set(Array.from(allowedHosts, (host) => normalizeHost(host.trim())).filter(Boolean));
}

async function validateTarget(url: URL, allowedHosts: ReadonlySet<string>, resolver: SafeImageResolver): Promise<ResolvedAddress[]> {
  if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new Error('Only HTTP(S) image URLs are supported.');
  if (url.username || url.password) throw new Error('Image URLs must not include credentials.');
  if (url.href.length > 2_048) throw new Error('Image URL is too long.');
  const hostname = normalizeHost(url.hostname);
  if (!hostname || !allowedHosts.has(hostname)) throw new Error('Image host is not on the configured allowlist.');
  const expectedPort = url.protocol === 'https:' ? '443' : '80';
  if (url.port && url.port !== expectedPort) throw new Error('Image URL uses a disallowed port.');
  if (isIP(hostname) && isBlockedAddress(hostname)) throw new Error('Image URL targets a blocked IP address.');
  const addresses = await resolver(hostname);
  if (addresses.length === 0 || addresses.some(({ address }) => isBlockedAddress(address))) {
    throw new Error('Image host resolves to a blocked or non-public IP address.');
  }
  return addresses;
}

function requestImageOnce(
  url: URL,
  addresses: readonly ResolvedAddress[],
  signal: AbortSignal,
  maxBytes: number,
): Promise<SafeImageResponse> {
  return new Promise((resolveResponse, rejectResponse) => {
    const transport = url.protocol === 'https:' ? httpsRequest : httpRequest;
    const lookupPinned: NonNullable<HttpRequestOptions['lookup']> = (_hostname, options, callback) => {
      const records = addresses.map((entry) => ({ address: entry.address, family: entry.family }));
      if (typeof options === 'object' && options !== null && 'all' in options && options.all) {
        (callback as (error: NodeJS.ErrnoException | null, addresses?: Array<{ address: string; family: number }>) => void)(null, records);
      } else {
        const first = records[0];
        (callback as (error: NodeJS.ErrnoException | null, address: string, family: number) => void)(null, first.address, first.family);
      }
    };
    const requestOptions = {
      protocol: url.protocol,
      hostname: normalizeHost(url.hostname),
      port: url.port || undefined,
      path: `${url.pathname}${url.search}`,
      method: 'GET',
      headers: { Accept: 'image/jpeg,image/png,image/webp' },
      lookup: lookupPinned,
      ...(url.protocol === 'https:' ? { servername: normalizeHost(url.hostname) } : {}),
    } as HttpRequestOptions & HttpsRequestOptions;

    const request = transport(requestOptions, (response) => {
      const contentLength = response.headers['content-length'];
      const contentLengthText = Array.isArray(contentLength) ? contentLength[0] : contentLength;
      if (contentLengthText && Number(contentLengthText) > maxBytes) {
        response.destroy(new Error('Image response exceeds the configured size limit.'));
        rejectResponse(new Error('Image response exceeds the configured size limit.'));
        return;
      }

      const statusCode = response.statusCode || 0;
      const locationHeader = response.headers.location;
      const contentTypeHeader = response.headers['content-type'];
      const responseContentType = Array.isArray(contentTypeHeader) ? contentTypeHeader[0] : contentTypeHeader;
      if (REDIRECT_STATUSES.has(statusCode)) {
        const location = Array.isArray(locationHeader) ? locationHeader[0] : locationHeader;
        response.destroy();
        resolveResponse({ statusCode, location, body: new Uint8Array() });
        return;
      }

      const chunks: Buffer[] = [];
      let totalBytes = 0;
      response.on('data', (chunk: Buffer | Uint8Array) => {
        totalBytes += chunk.byteLength;
        if (totalBytes > maxBytes) {
          response.destroy(new Error('Image response exceeds the configured size limit.'));
          request.destroy(new Error('Image response exceeds the configured size limit.'));
          rejectResponse(new Error('Image response exceeds the configured size limit.'));
          return;
        }
        chunks.push(Buffer.from(chunk));
      });
      response.on('end', () => resolveResponse({
        statusCode,
        contentType: responseContentType,
        contentLength: contentLengthText,
        body: Buffer.concat(chunks, totalBytes),
      }));
      response.on('error', rejectResponse);
    });

    const abortRequest = () => request.destroy(new Error('Image request timed out.'));
    if (signal.aborted) abortRequest();
    else signal.addEventListener('abort', abortRequest, { once: true });
    request.on('close', () => signal.removeEventListener('abort', abortRequest));
    request.on('error', rejectResponse);
    request.end();
  });
}

export async function fetchAllowedImage(
  input: string,
  options: SafeImageFetchOptions,
): Promise<{ body: Buffer; mimeType: string }> {
  const allowedHosts = normalizeAllowedHosts(options.allowedHosts);
  if (allowedHosts.size === 0) throw new Error('No remote image hosts are configured.');
  const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxRedirects = options.maxRedirects ?? DEFAULT_MAX_REDIRECTS;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new RangeError('maxBytes must be a positive integer.');
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) throw new RangeError('timeoutMs must be a positive integer.');
  if (!Number.isSafeInteger(maxRedirects) || maxRedirects < 0 || maxRedirects > 5) throw new RangeError('maxRedirects must be between 0 and 5.');

  const resolver = options.resolveHost ?? resolvePublicAddresses;
  const transport = options.transport ?? requestImageOnce;
  const abortController = new AbortController();
  const timeout = setTimeout(() => abortController.abort(), timeoutMs);
  let rejectTimeout!: (reason: Error) => void;
  const timeoutPromise = new Promise<never>((_resolve, reject) => {
    rejectTimeout = reject;
  });
  const onTimeout = () => rejectTimeout(new Error('Image request timed out.'));
  abortController.signal.addEventListener('abort', onTimeout, { once: true });
  let url: URL;

  try {
    url = new URL(input);
    for (let redirectCount = 0; ; redirectCount++) {
      const addresses = await Promise.race([
        validateTarget(url, allowedHosts, resolver),
        timeoutPromise,
      ]);
      if (abortController.signal.aborted) throw new Error('Image request timed out.');
      const response = await Promise.race([
        transport(url, addresses, abortController.signal, maxBytes),
        timeoutPromise,
      ]);

      if (REDIRECT_STATUSES.has(response.statusCode)) {
        if (!response.location || redirectCount >= maxRedirects) throw new Error('Image URL redirected too many times or omitted its destination.');
        url = new URL(response.location, url);
        continue;
      }
      if (response.statusCode < 200 || response.statusCode >= 300) throw new Error(`Image host returned HTTP ${response.statusCode}.`);
      const mimeType = (response.contentType || '').split(';', 1)[0].trim().toLowerCase();
      if (!ACCEPTED_IMAGE_TYPES.has(mimeType)) throw new Error('Image host returned an unsupported content type.');
      if (response.body.byteLength > maxBytes) throw new Error('Image response exceeds the configured size limit.');
      return { body: Buffer.from(response.body), mimeType };
    }
  } finally {
    clearTimeout(timeout);
    abortController.signal.removeEventListener('abort', onTimeout);
  }
}
