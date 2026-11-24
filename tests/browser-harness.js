// SPDX-License-Identifier: AGPL-3.0-only
import * as ort from 'onnxruntime-web/wasm';
import { loadModels } from '../src/models.js';
import { createPipeline } from '../src/pipeline.js';
import { createClient } from '../src/client.js';
import { classifyExpression } from '../src/expression.js';

const output = document.querySelector('#results');
const log = value => { output.textContent += value + '\n'; };
function check(ok, message) { if (!ok) throw new Error(message); }
function close(actual, expected, message, atol = 1e-4, rtol = 1e-4) {
  check(actual.length === expected.length, message + ' shape');
  actual.forEach((value, i) => check(Number.isFinite(value) && Math.abs(value - expected[i]) <= atol + rtol * Math.abs(expected[i]), message + ' index ' + i + ': ' + value + ' vs ' + expected[i]));
}
function rgb(size) {
  const values = new Float32Array(size * size * 3);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = y * size + x;
    values[i] = (13 * x + 7 * y + x * y % 251) % 256 / 255;
    values[size * size + i] = (3 * x + 17 * y + (x ^ y) % 239) % 256 / 255;
    values[2 * size * size + i] = (19 * x + 5 * y + x * y % 241) % 256 / 255;
  }
  return values;
}
document.querySelector('#run').addEventListener('click', async event => {
  event.target.disabled = true;
  output.textContent = '';
  let pipeline;
  const sessions = [];
  const client = createClient({ baseUrl: location.origin + '/' });
  try {
    const oracle = await fetch('./fixtures/oracle.json').then(response => response.json());
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = { mjs: new URL('/ort/1.31.1/ort-wasm-simd-threaded.mjs', location.href), wasm: new URL('/ort/1.31.1/ort-wasm-simd-threaded.wasm', location.href) };
    const bytes = await loadModels(location.origin + '/', new AbortController().signal, state => log('Loaded ' + state));
    pipeline = await createPipeline(ort, bytes);
    log('PASS all three real model contracts accepted');
    await pipeline.dispose(); pipeline = undefined;
    for (const key of ['faceDetector', 'expression', 'landmarks']) sessions.push(await ort.InferenceSession.create(bytes[key], { executionProviders: ['wasm'] }));
    for (const item of oracle.e1.examples) {
      const input = new Float32Array(48 * 48);
      for (let y = 0; y < 48; y++) for (let x = 0; x < 48; x++) input[y * 48 + x] =
        (item.name === 'checker3' ? (Math.floor(x / 3) + Math.floor(y / 3)) % 2 * 255 : item.name === 'mixed' ? (13 * x + 31 * y + x * y % 251) % 256 : item.name === 'ramp' ? (5 * x + 3 * y) % 256 : 13 * y % 256) / 255;
      const tensor = new ort.Tensor('float32', input, [1, 1, 48, 48]);
      const result = await sessions[1].run({ faces: tensor });
      close(result.logits.data, item.logits, 'E1 ' + item.name);
      check(classifyExpression(result.logits.data).label === item.expression.label, 'E1 accepted label ' + item.name);
      result.logits.dispose(); tensor.dispose(); input.fill(0);
    }
    log('PASS E1 four synthetic cases match CPU logits and calibrated labels');
    const meshInput = new ort.Tensor('float32', rgb(192), [1, 3, 192, 192]);
    const mesh = await sessions[2].run({ image: meshInput });
    close(mesh.landmarks.data, oracle.mesh.examples.pattern.landmarks_flat, 'mesh');
    close(mesh.scores.data, oracle.mesh.examples.pattern.scores, 'mesh score', 1e-5, 1e-3);
    check(mesh.scores.data[0] < .5, 'synthetic mesh must not be valid');
    Object.values(mesh).forEach(tensor => tensor.dispose()); meshInput.dispose();
    log('PASS 468 mesh points match CPU; synthetic input stays low_presence');
    const faceInput = new ort.Tensor('float32', rgb(640), [1, 3, 640, 640]);
    const detector = await sessions[0].run({ images: faceInput });
    close(detector.output0.data.slice(0, 72), oracle.yolo26_face.output_first_12_rows.flat(), 'YOLO', .02, 1e-3);
    check(!Array.from(detector.output0.data).some((value, i) => i % 6 === 4 && value >= .4), 'synthetic YOLO must have no faces');
    detector.output0.dispose(); faceInput.dispose();
    log('PASS YOLO rows match CPU; generated pattern has no faces');
    await Promise.all(sessions.map(session => session.release())); sessions.length = 0;
    await client.start();
    const result = await client.infer({ width: 64, height: 64, pixels: new Uint8ClampedArray(64 * 64 * 4) });
    check(result.faces.length === 0 && result.expressionStatus === 'no_faces', 'real worker no-face result');
    client.stop();
    await client.start(); client.stop();
    log('PASS real Worker loads three models, infers, stops and restarts');
    Object.values(bytes).forEach(value => value.fill(0));
    log('ALL CHECKS PASSED');
  } catch (error) { log('FAIL ' + error.stack); }
  finally { client.stop(); await pipeline?.dispose(); await Promise.allSettled(sessions.map(session => session.release())); event.target.disabled = false; }
});
