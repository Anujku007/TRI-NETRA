import { spawn } from 'node:child_process';

const port = process.env.PORT || 3000;
const url = `http://localhost:${port}`;

const server = spawn('npm run dev:server', { stdio: 'inherit', shell: true });
server.on('exit', (code) => process.exit(code ?? 0));
process.on('SIGINT', () => server.kill());

function openBrowser() {
  const win = process.platform === 'win32';
  const cmd = win ? 'cmd' : process.platform === 'darwin' ? 'open' : 'xdg-open';
  const args = win ? ['/c', 'start', '', url] : [url];
  spawn(cmd, args, { stdio: 'ignore', detached: true }).unref();
}

// wait until the server answers, then open the browser once
for (let i = 0; i < 120; i++) {
  try {
    await fetch(url, { signal: AbortSignal.timeout(1000) });
    openBrowser();
    break;
  } catch {
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}