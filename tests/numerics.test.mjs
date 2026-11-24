import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const oracle = JSON.parse(await readFile(new URL('./fixtures/oracle.json', import.meta.url)));
const digest = value => createHash('sha256').update(value).digest('hex');
const tensorDigest = value => digest(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
const box = values => Object.fromEntries(['x1', 'y1', 'x2', 'y2'].map((key, i) => [key, values[i]]));

function pattern(width, height) {
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const offset = (y * width + x) * 4;
    pixels.set([(13 * x + 7 * y + x * y % 251) % 256,
      (3 * x + 17 * y + (x ^ y) % 239) % 256,
      (19 * x + 5 * y + x * y % 241) % 256, 255], offset);
  }
  return { width, height, pixels };
}

function meshCase(item) {
  const points = new Float32Array(468 * 3);
  for (let i = 0; i < 468; i++) points.set(oracle.geometry.default_xyz, i * 3);
  for (const [index, xy] of Object.entries(oracle.geometry.layout_xy)) points.set(xy, Number(index) * 3);
  for (const [point, axis, value] of item.edits_point_axis_value) points[point * 3 + axis] = value === 'NaN' ? NaN : value;
  return points;
}

test('expression preprocessing matches independent Pillow bytes, tensor and crop quality', async () => {
  const { prepareExpression } = await import('../src/pixels.js');
  for (const item of oracle.preprocessing) {
    const result = prepareExpression(pattern(item.width, item.height), box(item.face_box));
    assert.equal(result.status, 'ready');
    assert.equal(digest(result.pixels), item.expression.u8_sha256);
    assert.deepEqual(result.pixels, new Uint8Array(Buffer.from(item.expression.u8_base64, 'base64')));
    assert.equal(tensorDigest(result.tensor), item.expression.tensor_sha256_le_f32);
  }
});

test('expression crops reject invalid, boundary and undersized faces before inference', async () => {
  const { prepareExpression } = await import('../src/pixels.js');
  const frame = pattern(127, 109);
  for (const [coordinates, status] of [
    [[NaN, 10, 90, 90], 'invalid_crop'], [[80, 10, 30, 90], 'invalid_crop'],
    [[0, 10, 90, 90], 'edge_crop'], [[10, 10, 127, 90], 'edge_crop'],
    [[10, 10, 49, 80], 'too_small'],
  ]) assert.equal(prepareExpression(frame, box(coordinates)).status, status);
});

test('flat and blurry crops produce truthful quality reasons', async () => {
  const { prepareExpression, qualityReason } = await import('../src/pixels.js');
  const pixels = new Uint8ClampedArray(100 * 100 * 4).fill(128);
  assert.equal(prepareExpression({ width: 100, height: 100, pixels }, box([10, 10, 90, 90])).status, 'low_contrast');
  const gradient = Uint8Array.from({ length: 48 * 48 }, (_, i) => i % 48 * 5);
  assert.equal(qualityReason(gradient, 48, 48), 'blurred');
});

test('mesh RGB and planar preprocessing match independent Python downsample and upsample', async () => {
  const { prepareMesh } = await import('../src/pixels.js');
  for (const item of oracle.preprocessing) {
    const result = prepareMesh(pattern(item.width, item.height), box(item.face_box));
    assert.equal(result.status, 'ready');
    assert.deepEqual(result.roi, item.mesh.square_roi_left_top_side);
    assert.equal(digest(result.pixels), item.mesh.rgb_u8_hwc_sha256);
    assert.equal(tensorDigest(result.tensor), item.mesh.tensor_sha256_le_f32);
  }
});

test('mesh rejects clipped and tiny ROIs while invalid frames never allocate model tensors', async () => {
  const { prepareMesh, prepareDetector } = await import('../src/pixels.js');
  const frame = pattern(127, 109);
  assert.equal(prepareMesh(frame, box([1, 1, 80, 80])).status, 'clipped_roi');
  assert.equal(prepareMesh(frame, box([10, 10, 61, 80])).status, 'too_small');
  for (const invalid of [
    { width: 0, height: 1, pixels: new Uint8ClampedArray(4) },
    { width: 641, height: 1, pixels: new Uint8ClampedArray(641 * 4) },
    { width: 2, height: 1, pixels: new Uint8ClampedArray(4) },
  ]) assert.throws(() => prepareDetector(invalid));
});

test('detector letterbox keeps RGB channel order and maps output back to a portrait frame', async () => {
  const { prepareDetector, decodeFaces } = await import('../src/pixels.js');
  const prepared = prepareDetector({ width: 1, height: 2, pixels: Uint8ClampedArray.of(255, 0, 0, 255, 255, 0, 0, 255) });
  assert.equal(prepared.tensor.length, 3 * 640 * 640);
  assert.equal(prepared.tensor[0], Math.fround(114 / 255));
  assert.equal(prepared.tensor[320], 1);
  assert.equal(prepared.tensor[640 * 640 + 320], 0);
  const values = new Float32Array(300 * 6);
  values.set([180, 50, 440, 100, .5, 0]);
  assert.deepEqual(decodeFaces({ type: 'float32', dims: [1, 300, 6], data: values }, 320, 640),
    [{ x1: 20, y1: 50, x2: 280, y2: 100, confidence: .5 }]);
  values[5] = 1;
  assert.throws(() => decodeFaces({ type: 'float32', dims: [1, 300, 6], data: values }, 320, 640));
});

test('frozen E1 calibration accepts/rejects real CPU oracle logits without hybrid changes', async () => {
  const { classifyExpression } = await import('../src/expression.js');
  for (const sample of oracle.e1.examples) {
    const result = classifyExpression(sample.logits);
    assert.equal(result.label, sample.accepted ? sample.winner : 'Unknown');
    assert.equal(result.confidence === null, !sample.accepted);
    result.probabilities.forEach((value, i) => assert.ok(Math.abs(value - sample.calibrated[i]) < 1e-12));
  }
});

test('expression malformed logits cannot become a valid-looking label', async () => {
  const { classifyExpression } = await import('../src/expression.js');
  for (const values of [[], [1, 2], [0, 0, NaN, 0, 0, 0, 0], [0, 0, Infinity, 0, 0, 0, 0]]) {
    assert.throws(() => classifyExpression(values));
  }
  assert.equal(classifyExpression([0, 0, 0, 0, 0, 0, 0]).label, 'Unknown');
});

test('468-point geometry matches Python values and rejects off-axis/clipped/nonfinite points', async () => {
  const { measureGeometry } = await import('../src/geometry.js');
  for (const item of Object.values(oracle.geometry.cases)) {
    const result = measureGeometry(meshCase(item));
    assert.equal(result.status, item.status);
    if (item.values) item.values.forEach((value, i) => assert.ok(Math.abs(result.values[i] - value) < 2e-6));
    else assert.equal(result.values, null);
  }
});

test('mesh low presence and malformed output never expose invented geometry', async () => {
  const { evaluateMesh } = await import('../src/geometry.js');
  const points = meshCase(oracle.geometry.cases.upright);
  assert.equal(evaluateMesh(.49, points).status, 'low_presence');
  assert.equal(evaluateMesh(.5, points).status, 'valid');
  assert.equal(evaluateMesh(NaN, points).status, 'invalid_output');
  assert.equal(evaluateMesh(1.1, points).status, 'invalid_output');
  assert.equal(evaluateMesh(.9, new Float32Array(3)).status, 'invalid_output');
});

test('quality gate preserves the NumPy float32 contrast boundary', async () => {
  const { qualityReason } = await import('../src/pixels.js');
  const pixels = Uint8Array.from({ length: 48 * 48 }, (_, i) => i % 2 ? 137 : 119);
  pixels.set([112, 128, 130, 136]);
  assert.equal(qualityReason(pixels, 48, 48), null);
  assert.equal(qualityReason(Uint8Array.from({ length: 2304 }, (_, i) => i % 2 ? 136 : 120), 48, 48), 'low_contrast');
  assert.equal(qualityReason(Uint8Array.from({ length: 2304 }, (_, i) => i % 2 ? 138 : 118), 48, 48), null);
  // Independent NumPy float32 Laplacian variances: 17.989601135253906 / 18.019847869873047.
  for (const [count, reason] of [[680, 'blurred'], [681, null]]) {
    const ramp = Uint8Array.from({ length: 2304 }, (_, i) => {
      const x = i % 48; const y = Math.floor(i / 48);
      return 10 + 3 * x + (i < count ? (x + y) % 2 ? 1 : -1 : 0);
    });
    assert.equal(qualityReason(ramp, 48, 48), reason);
  }
  assert.throws(() => qualityReason(new Uint8Array(0), 48, 48), /quality/i);
});
