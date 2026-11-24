import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { createServer, request } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

async function fixture(t, corrupt = false) {
  const { localModelsPlugin } = await import('../tools/dev-models.mjs');
  const directory = await mkdtemp(path.join(tmpdir(), 'core-cv-fixtures-'));
  const content = Uint8Array.of(1, 2, 3);
  const pins = Object.fromEntries(Object.entries({ faceDetector: 'yolo26-face-v1', expression: 'e1-gray48-v1', landmarks: 'facemesh468-v1' })
    .map(([name, contract]) => [name, { source: `${name}.onnx`, file: `${name.toLowerCase()}-v1.onnx`, contract,
      bytes: content.length, sha256: createHash('sha256').update(content).digest('hex') }]));
  for (const [name, pin] of Object.entries(pins)) await writeFile(path.join(directory, pin.source), corrupt && name === 'expression' ? Uint8Array.of(1, 2, 4) : content);
  let handler;
  localModelsPlugin({ directory, pins }).configureServer({ config: { base: '/' }, middlewares: { use(fn) { handler = fn; } } });
  const server = createServer((request, response) => handler(request, response, () => response.writeHead(404).end()));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  t.after(async () => {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    assert.ok(directory.startsWith(path.join(tmpdir(), 'core-cv-fixtures-')));
    await rm(directory, { recursive: true });
  });
  return { base, content, pins };
}

test('loopback-only dev route serves exactly verified bytes and a three-model manifest', async t => {
  const { base, content, pins } = await fixture(t);
  const manifest = await fetch(base + '/models/core-cv.json').then(response => response.json());
  assert.equal(manifest.schemaVersion, 1);
  assert.equal(Object.keys(manifest.models).length, 3);
  for (const [name, pin] of Object.entries(pins)) {
    assert.equal(manifest.models[name].sha256, pin.sha256);
    const response = await fetch(base + '/models/' + pin.file);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('cache-control'), 'no-store');
    assert.deepEqual(new Uint8Array(await response.arrayBuffer()), content);
  }
});

test('dev model route denies foreign Origin, Host, cross-site requests and mutations', async t => {
  const { base } = await fixture(t);
  for (const headers of [{ Origin: 'https://attacker.example' }, { Host: 'attacker.example' }, { 'Sec-Fetch-Site': 'cross-site' }]) {
    const status = await new Promise((resolve, reject) => {
      const outgoing = request(base + '/models/core-cv.json', { headers }, response => { response.resume(); resolve(response.statusCode); });
      outgoing.on('error', reject);
      outgoing.end();
    });
    assert.equal(status, 403);
  }
  const response = await fetch(base + '/models/core-cv.json', { method: 'POST' });
  assert.equal(response.status, 405);
  await response.arrayBuffer();
});

test('corrupt local weights stay unavailable and are not exposed as verified assets', async t => {
  const { base, pins } = await fixture(t, true);
  const manifest = await fetch(base + '/models/core-cv.json').then(response => response.json());
  assert.equal(manifest.models.expression, null);
  const response = await fetch(base + '/models/' + pins.expression.file);
  assert.equal(response.status, 404);
  await response.arrayBuffer();
});
