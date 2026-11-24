import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const oracle = JSON.parse(await readFile(new URL('./fixtures/oracle.json', import.meta.url)));
function meshPoints() {
  const values = new Float32Array(468 * 3);
  for (let i = 0; i < 468; i++) values.set(oracle.geometry.default_xyz, i * 3);
  for (const [key, xy] of Object.entries(oracle.geometry.layout_xy)) values.set(xy, Number(key) * 3);
  return values;
}
function frame() {
  const pixels = new Uint8ClampedArray(200 * 200 * 4);
  for (let y = 0; y < 200; y++) for (let x = 0; x < 200; x++) {
    const value = (Math.floor(x / 3) + Math.floor(y / 3)) % 2 * 255;
    pixels.set([value, value, value, 255], (y * 200 + x) * 4);
  }
  return { width: 200, height: 200, pixels };
}
function runtime({ failureAt = -1, badExpression = false, stallFace = false, faces = 1 } = {}) {
  const sessions = [];
  let releaseFace;
  class Tensor {
    constructor(type, data, dims) { Object.assign(this, { type, data, dims, disposed: false }); }
    dispose() { this.disposed = true; }
  }
  const signatures = [
    { input: ['images', [1, 3, 640, 640]], outputs: [['output0', [1, 300, 6]]] },
    { input: ['faces', ['batch', 1, 48, 48]], outputs: [['logits', ['batch', 7]]] },
    { input: ['image', [1, 3, 192, 192]], outputs: [['scores', [1]], ['landmarks', [1, 468, 3]]] },
  ];
  const meta = ([name, shape]) => ({ name, shape, type: 'float32', isTensor: true });
  const ort = { Tensor, InferenceSession: { async create() {
    const index = sessions.length;
    if (index === failureAt) throw new Error('Model initialization failed');
    const signature = signatures[index];
    const session = { released: 0, inputMetadata: [meta(signature.input)], outputMetadata: signature.outputs.map(meta),
      async release() { this.released++; }, async run(inputs) {
        if (index === 0) {
          if (stallFace) await new Promise(resolve => { releaseFace = resolve; });
          const data = new Float32Array(1800);
          for (let i = 0; i < faces; i++) data.set([64 + i * 6, 64, 384 + i * 6, 384, .9, 0], i * 6);
          return { output0: new Tensor('float32', data, [1, 300, 6]) };
        }
        if (index === 1) {
          if (badExpression) throw new Error('CNN failed');
          const count = inputs.faces.dims[0];
          const data = new Float32Array(count * 7);
          for (let i = 0; i < count; i++) data.set(oracle.e1.examples[0].logits, i * 7);
          return { logits: new Tensor('float32', data, [count, 7]) };
        }
        return { scores: new Tensor('float32', Float32Array.of(.9), [1]),
          landmarks: new Tensor('float32', meshPoints(), [1, 468, 3]) };
      } };
    sessions.push(session);
    return session;
  } } };
  return { ort, sessions, releaseFace: () => releaseFace?.() };
}
const bytes = { faceDetector: Uint8Array.of(1), expression: Uint8Array.of(2), landmarks: Uint8Array.of(3) };

test('three-model pipeline returns real calibrated classifications and mesh geometry aligned to detected faces', async () => {
  const { createPipeline } = await import('../src/pipeline.js');
  const fixture = runtime();
  const pipeline = await createPipeline(fixture.ort, bytes);
  try {
    const result = await pipeline.infer(frame(), 1000);
    assert.equal(result.faces.length, 1);
    assert.equal(result.faces[0].expression.label, 'Angry');
    assert.equal(result.faces[0].geometry.status, 'valid');
    assert.equal(result.faces[0].geometry.values.length, 12);
    assert.equal(result.expressionStatus, 'sampled');
    assert.equal(result.landmarkStatus, 'sampled');
    const next = await pipeline.infer(frame(), 1250);
    assert.equal(next.landmarkStatus, 'skipped');
    assert.equal(next.faces[0].geometry.values, null);
  } finally { await pipeline.dispose(); }
  assert.deepEqual(fixture.sessions.map(item => item.released), [1, 1, 1]);
});

test('initialization failure releases previously opened sessions and rejects partial readiness', async () => {
  const { createPipeline } = await import('../src/pipeline.js');
  const fixture = runtime({ failureAt: 2 });
  await assert.rejects(createPipeline(fixture.ort, bytes));
  assert.deepEqual(fixture.sessions.map(item => item.released), [1, 1]);
});

test('CNN failure stays Unknown while detector and geometry continue truthfully', async () => {
  const { createPipeline } = await import('../src/pipeline.js');
  const fixture = runtime({ badExpression: true });
  const pipeline = await createPipeline(fixture.ort, bytes);
  try {
    const result = await pipeline.infer(frame(), 1000);
    assert.equal(result.expressionStatus, 'error');
    assert.equal(result.faces[0].expression.label, 'Unknown');
    assert.equal(result.faces[0].expression.confidence, null);
    assert.equal(result.faces[0].geometry.status, 'valid');
  } finally { await pipeline.dispose(); }
});

test('face budgets keep per-face alignment and do not classify extra faces', async () => {
  const { createPipeline } = await import('../src/pipeline.js');
  const fixture = runtime({ faces: 6 });
  const pipeline = await createPipeline(fixture.ort, bytes);
  try {
    const result = await pipeline.infer(frame(), 1000);
    assert.equal(result.faces.length, 6);
    assert.equal(result.faces[4].expression.reason, 'face_budget');
    assert.equal(result.faces[5].expression.label, 'Unknown');
    assert.equal(result.faces[2].geometry.status, 'face_budget');
  } finally { await pipeline.dispose(); }
});

test('pipeline rejects overlapping inference and use after disposal', async () => {
  const { createPipeline } = await import('../src/pipeline.js');
  const fixture = runtime({ stallFace: true });
  const pipeline = await createPipeline(fixture.ort, bytes);
  const first = pipeline.infer(frame(), 1000);
  await assert.rejects(pipeline.infer(frame(), 1100), /busy/i);
  fixture.releaseFace();
  await first;
  await pipeline.dispose();
  await pipeline.dispose();
  await assert.rejects(pipeline.infer(frame(), 1400), /closed/i);
  assert.deepEqual(fixture.sessions.map(item => item.released), [1, 1, 1]);
});
