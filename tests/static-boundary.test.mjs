import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

test('Vite cannot serve models or sibling private files through its static fallback', async t => {
  const { createServer } = await import('vite');
  const { default: config } = await import('../vite.config.js');
  const root = await mkdtemp(path.join(tmpdir(), 'core-cv-static-test-'));
  const inside = path.join(root, 'app'); await mkdir(inside);
  await mkdir(path.join(inside, 'local-models'));
  await writeFile(path.join(inside, 'local-models/unverified.onnx'), Uint8Array.of(1, 2, 3));
  await writeFile(path.join(inside, 'unverified.onnx'), Uint8Array.of(1, 2, 3));
  await writeFile(path.join(inside, 'safe.txt'), 'public static control');
  await writeFile(path.join(root, 'private.txt'), 'private');
  const server = await createServer({ ...config, root: inside, configFile: false, logLevel: 'silent',
    server: { ...config.server, port: 0, strictPort: false, fs: { ...config.server.fs, allow: [inside] } } });
  await server.listen();
  t.after(async () => { await server.close(); assert.ok(root.startsWith(path.join(tmpdir(), 'core-cv-static-test-'))); await rm(root, { recursive: true }); });
  const base = `http://127.0.0.1:${server.httpServer.address().port}`;
  assert.equal((await fetch(base + '/safe.txt', { method: 'HEAD' })).status, 200);
  for (const route of ['/local-models/unverified.onnx', '/unverified.onnx', '/@fs/' + path.join(root, 'private.txt').replaceAll('\\', '/')]) {
    const response = await fetch(base + route, { method: 'HEAD' });
    assert.notEqual(response.status, 200, route);
  }
});
