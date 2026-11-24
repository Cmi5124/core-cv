// SPDX-License-Identifier: AGPL-3.0-only
import { createHash } from 'node:crypto';
import { open, realpath } from 'node:fs/promises';
import path from 'node:path';
import { MODEL_PINS } from './model-specs.mjs';

export function localModelsPlugin({ directory, pins = MODEL_PINS }) {
  const verified = new Map();
  async function read(pin) {
    if (verified.has(pin.file)) return verified.get(pin.file);
    if (!/^[a-zA-Z0-9_.-]+\.onnx$/.test(pin.source) || !/^[a-zA-Z0-9_.-]+\.onnx$/.test(pin.file) ||
        !Number.isSafeInteger(pin.bytes) || pin.bytes <= 0 || pin.bytes > 16 * 1024 * 1024 || !/^[a-f0-9]{64}$/.test(pin.sha256)) return null;
    let handle;
    try {
      const root = await realpath(directory);
      const filename = await realpath(path.join(root, pin.source));
      if (path.dirname(filename) !== root) return null;
      handle = await open(filename, 'r');
      const stat = await handle.stat();
      if (!stat.isFile() || stat.size !== pin.bytes) return null;
      const bytes = Buffer.alloc(pin.bytes);
      let offset = 0;
      while (offset < bytes.length) {
        const result = await handle.read(bytes, offset, bytes.length - offset, offset);
        if (!result.bytesRead) return null;
        offset += result.bytesRead;
      }
      if (createHash('sha256').update(bytes).digest('hex') !== pin.sha256) return null;
      verified.set(pin.file, bytes);
      return bytes;
    } catch { return null; }
    finally { await handle?.close(); }
  }
  return {
    name: 'core-cv-loopback-models',
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const pathname = request.url?.split('?')[0];
        if (pathname !== '/models/core-cv.json' && !Object.values(pins).some(pin => pathname === '/models/' + pin.file)) return next();
        const host = request.headers.host ?? '';
        const local = ['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(request.socket.remoteAddress);
        if (!local || !/^(127\.0\.0\.1|localhost)(:\d{1,5})?$/.test(host) ||
            request.headers.origin && request.headers.origin !== `http://${host}` ||
            request.headers['sec-fetch-site'] && !['same-origin', 'none'].includes(request.headers['sec-fetch-site'])) {
          response.writeHead(403).end(); return;
        }
        if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405, { Allow: 'GET, HEAD' }).end(); return; }
        response.setHeader('Cache-Control', 'no-store');
        response.setHeader('X-Content-Type-Options', 'nosniff');
        if (pathname === '/models/core-cv.json') {
          const models = {};
          for (const [name, pin] of Object.entries(pins)) models[name] = await read(pin)
            ? { file: pin.file, contract: pin.contract, bytes: pin.bytes, sha256: pin.sha256 } : null;
          response.setHeader('Content-Type', 'application/json');
          response.end(request.method === 'HEAD' ? undefined : JSON.stringify({ schemaVersion: 1, models }));
        } else {
          const pin = Object.values(pins).find(item => pathname === '/models/' + item.file);
          const bytes = await read(pin);
          if (!bytes) { response.writeHead(404).end(); return; }
          response.setHeader('Content-Type', 'application/octet-stream');
          response.setHeader('Content-Length', bytes.length);
          response.end(request.method === 'HEAD' ? undefined : bytes);
        }
      });
    },
  };
}
