# Trinetra

Trinetra is a browser based surveillance dashboard with deterministic simulation scenarios, local face detection, optional Gemini analysis, incident logging, and phone camera streaming.

## Requirements

- Node.js 20 or newer
- npm
- A modern browser with camera permissions for webcam use

## Run locally

```sh
npm install
cp .env.example .env
npm run dev
```

Open the URL printed by the server (normally `http://localhost:3000`). Simulation scenarios and incident management work without a Gemini key. For live AI analysis, add a valid `GEMINI_API_KEY` to `.env`; without one, live feeds use the bundled local face detector and unmatched faces are marked for manual review.

## Presentation demo

For a clean, repeatable run that keeps demo records separate from the normal runtime files, start the development server with temporary storage:

```sh
mkdir -p /tmp/trinetra-demo
PORT=3100 DISABLE_HMR=true \
TRINETRA_LOCAL_DATA_DIR=/tmp/trinetra-demo/local \
TRINETRA_PERSONNEL_FILE=/tmp/trinetra-demo/personnel.json \
TRINETRA_TRACKING_FILE=/tmp/trinetra-demo/tracking.json \
TRINETRA_UNAUTHORIZED_INCIDENTS_FILE=/tmp/trinetra-demo/incidents.json \
npm run dev
```

Open `http://localhost:3100`, sign in with the configured `ADMIN_PASSWORD` (or the development default `admin` / `admin123` when none is set), then:

1. Open **Personnel Directory** and select **Load synthetic demo profiles**. These fictional entries have no photos or biometric templates.
2. Return to **Live**, choose a **Simulation** scenario, and let the scan complete. The illustration is generated locally; the event outcome is deterministic playback, not a live detector result or accuracy measurement.
3. Open **Incidents** to acknowledge, resolve, inspect, and clear the sample event. The temporary incident file is under `/tmp/trinetra-demo`.
4. For a live face-detection or tracking demonstration, use a camera and enrollment photo only from a participant who has explicitly consented. Simulation scenarios intentionally do not create live presence records.

Phone-camera pairing can be demonstrated with the admin QR flow. A remote phone needs a reachable HTTPS address; a phone cannot open the laptop's `localhost` address.

## Project structure

```text
frontend/             React application and Vite entry page
  src/                Components, screens, data, and browser utilities
backend/              Express API and server-side services
  index.ts            Application server entrypoint
  models/             Local face and hand detection models
  scripts/            Camera and maintenance scripts
  services/           Server-side face analysis
  .local/             Private runtime history and admin sessions
shared/               Types and utilities used by both sides
dist/                 Generated production build
```

The development admin login is `admin` / `admin123`. It is available only in development mode. Production requires a unique `ADMIN_PASSWORD` of at least 8 characters; change it in the Admin dialog before using the app outside a local development session.

## Production build

Set a unique `ADMIN_PASSWORD` (at least 8 characters) in the production environment. Optionally set `GEMINI_API_KEY` for Gemini analysis, then run:

```sh
npm run build
npm start
```

The server listens on port `3000` by default; set `PORT` to change it. The first successful admin password change is stored as a salted hash in `backend/.local/admin-password.json`. Keep that file private and back it up with the deployment configuration.

Authorized-person location sightings and daily camera transitions are saved in `backend/.local/tracking.json` so history survives server restarts. Keep this file private; set `TRINETRA_TRACKING_FILE` to choose a different storage path.

The authorized personnel directory and enrolled face templates are saved in `backend/.local/personnel.json`. New registrations, directory edits, and removals are synchronized to this file and restored when the server starts. Keep it private; set `TRINETRA_PERSONNEL_FILE` to choose a different storage path.

Biometric-breach and identity-review incidents are saved in `backend/.local/unauthorized-incidents.json`, with photos in the adjacent `unauthorized-evidence/` directory. Records and photos older than `TRINETRA_INCIDENT_RETENTION_DAYS` (default 90) and records beyond `TRINETRA_INCIDENT_MAX_RECORDS` (default 1,000) are deleted hourly, and whenever the incident store is read or written. Set either value to a positive integer to change the policy. Admins can also delete an incident or use **Clear Logs** for immediate deletion. Keep this data directory on persistent storage; set `TRINETRA_UNAUTHORIZED_INCIDENTS_FILE` to choose a different database file.

## Security controls

- Personnel profiles and biometric templates, tracking history, incident records and evidence photos, and surveillance analysis APIs require an active admin bearer session. Phone-camera and WebRTC APIs require an admin session or a token scoped to that camera pairing; pairing tokens expire after 8 hours, active MJPEG connections close at authorization expiry, and tokens can be revoked in the server API.
- Server-side remote image fetching is disabled unless `TRINETRA_ALLOWED_IMAGE_HOSTS` lists exact hostnames separated by commas (for example, `images.example.org,cdn.example.org`). Wildcards are not accepted. The fetcher blocks private, loopback, link-local, reserved, and non-public DNS results, revalidates every redirect, pins requests to the checked DNS addresses, accepts only JPEG/PNG/WebP, limits responses to 8 MiB by default, and applies a 4-second default timeout. Specific analysis paths may use smaller limits. Unsplash profile-photo registration is separately restricted to `images.unsplash.com`.
- Remote image analysis cannot fetch private-network camera URLs from the server. Configure a public allowlisted host or use a browser-accessible camera source that does not require server-side fetching.
- Incident evidence is stored under the same private runtime data directory as the incident index, with restrictive file permissions where supported. Browser-side legacy personnel, incident, and photo caches are cleared on application startup and logout.

Automated security checks run with `npm test`; `npm run lint` type-checks the application and `npm run build` creates the production bundles.

## Camera sources

- **Simulation:** Runs the bundled scenarios without a camera or API key.
- **Webcam:** Requires a browser camera permission. Browsers generally allow camera access on `localhost` or an HTTPS origin.
- **Upload:** Accepts JPG and PNG snapshots up to 10 MB.
- **Phone stream:** Open the pairing link on the phone and grant camera permission. Non-local deployments need HTTPS for browser camera access.
- **IP stream:** Configure a direct image or snapshot URL supported by the camera. Video stream formats vary by camera vendor.
- **Bluetooth:** Browser Bluetooth provides device metadata, not camera video. Connect an image source separately to analyze frames.

Lockdown mode, sirens, incident logs, and dispatch requests are local dashboard actions. This app does not control physical access systems or contact an external response team.
