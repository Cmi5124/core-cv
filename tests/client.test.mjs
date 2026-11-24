import assert from 'node:assert/strict';
import test from 'node:test';
import { E1_CHECKPOINT } from '../src/expression.js';

class WorkerDouble extends EventTarget {
  sent = [];
  terminated = 0;
  postMessage(value) { this.sent.push(value); }
  terminate() { this.terminated++; }
  reply(value) { this.dispatchEvent(new MessageEvent('message', { data: value })); }
}
const pixels = () => ({ width: 1, height: 1, pixels: Uint8ClampedArray.of(0, 0, 0, 255) });
const result = () => ({ width: 1, height: 1, faces: [], expressionStatus: 'no_faces', landmarkStatus: 'no_faces', inferenceMs: 1, modelVersion: E1_CHECKPOINT });

test('worker client prepares all models before inferring and rejects overlapping frames', async () => {
  const { createClient } = await import('../src/client.js');
  const worker = new WorkerDouble();
  const client = createClient({ baseUrl: 'https://core.example/', createWorker: () => worker });
  const opening = client.start();
  await assert.rejects(client.infer(pixels()), /ready/i);
  worker.reply({ id: worker.sent[0].id, type: 'ready' });
  await opening;
  const first = client.infer(pixels());
  await assert.rejects(client.infer(pixels()), /busy/i);
  worker.reply({ id: worker.sent[1].id, type: 'result', result: result() });
  assert.deepEqual(await first, result());
  client.stop();
  assert.equal(worker.terminated, 1);
});

test('closing initialization rejects promptly and a late ready cannot restore the old worker', async () => {
  const { createClient } = await import('../src/client.js');
  const workers = [new WorkerDouble(), new WorkerDouble()];
  let next = 0;
  const client = createClient({ baseUrl: 'https://core.example/', createWorker: () => workers[next++] });
  const opening = client.start();
  client.stop();
  workers[0].reply({ id: workers[0].sent[0].id, type: 'ready' });
  await assert.rejects(opening, /stopped/i);
  const reopening = client.start();
  workers[0].reply({ id: workers[0].sent[0].id, type: 'ready' });
  workers[1].reply({ id: workers[1].sent[0].id, type: 'ready' });
  await reopening;
  client.stop();
  assert.deepEqual(workers.map(worker => worker.terminated), [1, 1]);
});

test('abort during a frame terminates the worker and stale results stay discarded', async () => {
  const { createClient } = await import('../src/client.js');
  const worker = new WorkerDouble();
  const client = createClient({ baseUrl: 'https://core.example/', createWorker: () => worker });
  const opening = client.start();
  worker.reply({ id: worker.sent[0].id, type: 'ready' });
  await opening;
  const controller = new AbortController();
  const pending = client.infer(pixels(), controller.signal);
  controller.abort(new DOMException('Stopped by user', 'AbortError'));
  worker.reply({ id: worker.sent[1].id, type: 'result', result: result() });
  await assert.rejects(pending, error => error.name === 'AbortError');
  assert.equal(worker.terminated, 1);
  await assert.rejects(client.infer(pixels()), /ready/i);
});

test('a bounded load deadline cleans up a stalled worker', async () => {
  const { createClient } = await import('../src/client.js');
  const worker = new WorkerDouble();
  const client = createClient({ baseUrl: 'https://core.example/', createWorker: () => worker, loadTimeoutMs: 15 });
  await assert.rejects(client.start(), /load_timeout/i);
  assert.equal(worker.terminated, 1);
});

test('malformed worker results are rejected and terminate their worker', async () => {
  const { createClient } = await import('../src/client.js');
  const worker = new WorkerDouble();
  const client = createClient({ baseUrl: 'https://core.example/', createWorker: () => worker });
  const opening = client.start();
  worker.reply({ id: worker.sent[0].id, type: 'ready' });
  await opening;
  const pending = client.infer(pixels());
  worker.reply({ id: worker.sent[1].id, type: 'result', result: { ...result(), width: 999 } });
  await assert.rejects(pending, /invalid_response/i);
  assert.equal(worker.terminated, 1);
});

test('abort while creating a worker also releases it', async () => {
  const { createClient } = await import('../src/client.js');
  const worker = new WorkerDouble();
  const controller = new AbortController();
  const client = createClient({ baseUrl: 'https://core.example/', createWorker() { controller.abort(); return worker; } });
  await assert.rejects(client.start({ signal: controller.signal }), error => error.name === 'AbortError');
  assert.equal(worker.terminated, 1);
});
