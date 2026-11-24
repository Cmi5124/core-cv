import assert from 'node:assert/strict';
import test from 'node:test';

async function fixture(overrides = {}) {
  const { createSession } = await import('../src/session.js');
  const events = []; let stopCount = 0; let scheduled;
  const camera = { stop() { stopCount++; } };
  const client = { async start() {}, async infer() { return { faces: [] }; }, stop() {} };
  const session = createSession({ client, openCamera: async () => camera,
    readFrame: () => ({ width: 1, height: 1, pixels: new Uint8ClampedArray(4) }),
    onState: value => events.push(value), onResult: value => events.push(value),
    schedule: callback => { scheduled = callback; return 1; }, cancel() { scheduled = undefined; }, ...overrides });
  return { session, events, camera, stopped: () => stopCount, tick: () => scheduled?.() };
}

test('consent starts all models then camera, one frame at a time; stop clears capture and UI', async () => {
  const f = await fixture();
  await f.session.start();
  assert.equal(f.events[0].state, 'loading');
  assert.ok(f.events.some(event => event.state === 'running'));
  await f.tick();
  assert.ok(f.events.some(event => Array.isArray(event.faces)));
  f.session.stop();
  assert.equal(f.stopped(), 1);
  assert.equal(f.events.at(-1).state, 'stopped');
});

test('late camera permission after stop cannot restart capture and its tracks are released', async () => {
  let accept; let stopped = 0;
  const f = await fixture({ openCamera: () => new Promise(resolve => { accept = resolve; }) });
  const opening = f.session.start();
  await new Promise(resolve => setImmediate(resolve));
  f.session.stop();
  accept({ stop() { stopped++; } });
  await opening;
  assert.equal(stopped, 1);
  assert.ok(!f.events.some(event => event.state === 'running'));
});

test('missing models never opens camera; permission denial and inference failure release resources', async t => {
  for (const stage of ['model', 'camera', 'frame']) await t.test(stage, async () => {
    let cameraRequests = 0; let workerStops = 0;
    const f = await fixture({ client: { async start() { if (stage === 'model') throw new Error('models_unpublished'); },
      async infer() { throw new Error('inference_timeout'); }, stop() { workerStops++; } },
      openCamera: async () => { cameraRequests++; if (stage === 'camera') throw new DOMException('Denied', 'NotAllowedError'); return { stop() {} }; } });
    await f.session.start();
    if (stage === 'frame') await f.tick();
    assert.equal(cameraRequests, stage === 'model' ? 0 : 1);
    assert.equal(f.events.at(-1).state, 'error');
    assert.equal(f.events.at(-1).code, stage === 'model' ? 'models_unpublished' : stage === 'camera' ? 'NotAllowedError' : 'inference_timeout');
    assert.equal(f.events.at(-1).stage, stage === 'model' ? 'model' : stage === 'camera' ? 'camera' : 'inference');
    assert.ok(workerStops >= 1);
  });
});

test('result arriving after revoke is discarded', async () => {
  let finish;
  const f = await fixture({ client: { async start() {}, infer: () => new Promise(resolve => { finish = resolve; }), stop() {} } });
  await f.session.start();
  const frame = f.tick();
  f.session.stop();
  finish({ faces: ['late'] });
  await frame;
  assert.ok(!f.events.some(event => event.faces));
  assert.equal(f.stopped(), 1);
});
