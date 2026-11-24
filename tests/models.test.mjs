import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';

const base = 'https://core.example/';
const contracts = { faceDetector: 'yolo26-face-v1', expression: 'e1-gray48-v1', landmarks: 'facemesh468-v1' };

function assets(t, change = value => value) {
  const original = globalThis.location;
  globalThis.location = { origin: 'https://core.example', href: base };
  t.after(() => { if (original === undefined) delete globalThis.location; else globalThis.location = original; });
  const bytes = Uint8Array.of(1, 2, 3);
  const models = Object.fromEntries(Object.entries(contracts).map(([key, contract]) => [key, {
    file: `${key.toLowerCase()}.onnx`, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex'), contract,
  }]));
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url: String(url), options });
    return String(url).endsWith('core-cv.json') ? Response.json(change({ schemaVersion: 1, models })) : new Response(bytes);
  });
  return { bytes, calls, models };
}

test('loads all three same-origin assets with verified bytes and cache-safe requests', async t => {
  const { bytes, calls, models } = assets(t);
  const { loadModels } = await import('../src/models.js');
  const loaded = await loadModels(base, new AbortController().signal, undefined, models);
  assert.deepEqual(Object.keys(loaded), ['faceDetector', 'expression', 'landmarks']);
  for (const key of Object.keys(loaded)) assert.deepEqual(loaded[key], bytes);
  assert.equal(calls.length, 4);
  assert.equal(calls[0].options.cache, 'no-store');
  for (const call of calls) {
    assert.equal(new URL(call.url).origin, 'https://core.example');
    assert.equal(call.options.credentials, 'omit');
    assert.equal(call.options.redirect, 'error');
  }
  assert.equal(calls[1].options.cache, 'default');
});

test('unpublished manifests and invalid model contracts never fetch model bytes', async t => {
  const { loadModels } = await import('../src/models.js');
  for (const invalid of [null, { file: '../secret.onnx' }, { file: 'https://other.example/model.onnx' },
    { bytes: -1 }, { bytes: 40 * 1024 * 1024 }, { sha256: 'bad' }, { contract: 'other-classifier' }]) {
    await t.test(JSON.stringify(invalid), async sub => {
      const { calls, models } = assets(sub, manifest => ({ ...manifest, models: { ...manifest.models,
        expression: invalid === null ? null : { ...manifest.models.expression, ...invalid },
      } }));
      await assert.rejects(loadModels(base, new AbortController().signal, undefined, models));
      assert.equal(calls.length, 1);
    });
  }
});

test('cache content with the wrong digest, size, missing body or HTTP status is rejected', async t => {
  const { loadModels } = await import('../src/models.js');
  for (const payload of [null, [1, 2], [1, 2, 3, 4], [1, 2, 4]]) {
    await t.test(String(payload), async sub => {
      const fixture = assets(sub);
      sub.mock.method(globalThis, 'fetch', async url => String(url).endsWith('core-cv.json')
        ? Response.json({ schemaVersion: 1, models: fixture.models })
        : new Response(payload ? Uint8Array.from(payload) : null, { status: payload ? 200 : 404 }));
      await assert.rejects(loadModels(base, new AbortController().signal, undefined, fixture.models));
    });
  }
});

test('abort cancels a stalled model body rather than leaving a pending download', async t => {
  const { loadModels } = await import('../src/models.js');
  const fixture = assets(t);
  const controller = new AbortController();
  let cancelled = false;
  let begin;
  const started = new Promise(resolve => { begin = resolve; });
  t.mock.method(globalThis, 'fetch', async url => {
    if (String(url).endsWith('core-cv.json')) return Response.json({ schemaVersion: 1, models: fixture.models });
    const body = new ReadableStream({ start() { begin(); }, cancel() { cancelled = true; } });
    return new Response(body);
  });
  const pending = loadModels(base, controller.signal, undefined, fixture.models);
  await started;
  controller.abort(new DOMException('Stopped', 'AbortError'));
  await assert.rejects(pending, error => error.name === 'AbortError');
  assert.equal(cancelled, true);
});

test('a self-consistent substituted manifest cannot impersonate the frozen checkpoint', async t => {
  const { calls } = assets(t);
  const { loadModels } = await import('../src/models.js');
  await assert.rejects(loadModels(base, new AbortController().signal), error => error.code === 'model_profile_mismatch');
  assert.equal(calls.length, 1);
});
