import type { Express, Request, RequestHandler } from 'express';

export interface AdminSession {
  username: string;
  role: string;
  loginTime: string;
  expiresAt: number;
}

export type AdminSessionResolver = (request: Request) => AdminSession | null;
export type PairTokenResolver = (pairCode: string, token: string) => number | null;

export const ADMIN_PROTECTED_API_PREFIXES = [
  '/api/personnel',
  '/api/tracking',
  '/api/incidents',
  '/api/surveillance',
] as const;

export const PAIRED_CAMERA_API_PREFIXES = [
  '/api/phone-camera',
  '/api/webrtc',
] as const;

export function getBearerToken(header: string | undefined): string | null {
  if (!header) return null;
  const match = /^Bearer\s+([^\s]+)$/i.exec(header.trim());
  return match?.[1] || null;
}

export function getPairCodeFromRequest(request: Request): string | null {
  const pathname = new URL(request.originalUrl, 'http://localhost').pathname;
  const pathParts = pathname.split('/').filter(Boolean);
  const pathPairCode = pathParts[0] === 'api' &&
    (pathParts[1] === 'phone-camera' || pathParts[1] === 'webrtc')
    ? pathParts[3]
    : undefined;
  const bodyPairCode = request.body && typeof request.body.pairCode === 'string'
    ? request.body.pairCode
    : undefined;
  const headerPairCode = request.get('x-camera-pair-code');
  const candidate = pathPairCode || headerPairCode || bodyPairCode;
  if (!candidate || candidate.length > 128 || /[\u0000-\u001f\u007f]/.test(candidate)) return null;
  try {
    return decodeURIComponent(candidate).toUpperCase();
  } catch {
    return null;
  }
}

export function createRequireAdmin(resolveSession: AdminSessionResolver): RequestHandler {
  return (request, response, next) => {
    const session = resolveSession(request);
    if (!session || session.expiresAt <= Date.now()) {
      return response.status(401).json({ success: false, error: 'An active admin session is required.' });
    }
    response.locals.adminSession = session;
    return next();
  };
}

export function createRequireAdminOrPairToken(
  resolveSession: AdminSessionResolver,
  resolvePairToken: PairTokenResolver,
): RequestHandler {
  return (request, response, next) => {
    const session = resolveSession(request);
    if (session && session.expiresAt > Date.now()) {
      response.locals.adminSession = session;
      response.locals.authorizationExpiresAt = session.expiresAt;
      return next();
    }

    const token = getBearerToken(request.headers.authorization);
    const pairCode = getPairCodeFromRequest(request);
    const pairTokenExpiresAt = token && pairCode ? resolvePairToken(pairCode, token) : null;
    if (pairTokenExpiresAt && pairTokenExpiresAt > Date.now()) {
      response.locals.authorizationExpiresAt = pairTokenExpiresAt;
      return next();
    }
    return response.status(401).json({ success: false, error: 'An active admin session or paired camera token is required.' });
  };
}

export function installSensitiveApiAuthorization(
  app: Express,
  resolveSession: AdminSessionResolver,
  resolvePairToken: PairTokenResolver,
): void {
  app.use([...ADMIN_PROTECTED_API_PREFIXES], createRequireAdmin(resolveSession));
  app.use([...PAIRED_CAMERA_API_PREFIXES], createRequireAdminOrPairToken(resolveSession, resolvePairToken));
}
