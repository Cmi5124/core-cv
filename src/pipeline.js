// SPDX-License-Identifier: AGPL-3.0-only
import { decodeFaces, prepareDetector, prepareExpression, prepareMesh, validateFrame } from './pixels.js';
import { classifyExpression, E1_CHECKPOINT } from './expression.js';
import { evaluateMesh } from './geometry.js';

const unknown = reason => ({ label: 'Unknown', confidence: null, reason });
const emptyGeometry = status => ({ status, values: null });
const shapeIs = (actual, expected) => Array.isArray(actual) && actual.length === expected.length && expected.every((value, i) => value === actual[i]);

function validateSession(session, inputName, inputShape, outputs) {
  const input = session.inputMetadata?.[0];
  if (session.inputMetadata?.length !== 1 || input.name !== inputName || !input.isTensor || input.type !== 'float32' ||
      !shapeIs(input.shape, inputShape) || session.outputMetadata?.length !== outputs.length ||
      outputs.some(([name, shape], i) => {
        const output = session.outputMetadata[i];
        return !output?.isTensor || output.name !== name || output.type !== 'float32' || !shapeIs(output.shape, shape);
      })) throw new Error('Invalid model tensor contract');
}

export async function createPipeline(ort, bytes) {
  const sessions = [];
  let closed = false;
  let busy = false;
  let lastExpression = -Infinity;
  let lastMesh = -Infinity;
  async function dispose() {
    if (closed) return;
    closed = true;
    await Promise.allSettled(sessions.map(session => session.release()));
  }
  try {
    for (const name of ['faceDetector', 'expression', 'landmarks']) {
      if (!(bytes[name] instanceof Uint8Array) || !bytes[name].length) throw new Error('Missing model bytes');
      sessions.push(await ort.InferenceSession.create(bytes[name], { executionProviders: ['wasm'] }));
    }
    validateSession(sessions[0], 'images', [1, 3, 640, 640], [['output0', [1, 300, 6]]]);
    validateSession(sessions[1], 'faces', ['batch', 1, 48, 48], [['logits', ['batch', 7]]]);
    validateSession(sessions[2], 'image', [1, 3, 192, 192], [['scores', [1]], ['landmarks', [1, 468, 3]]]);
  } catch (error) { await dispose(); throw error; }

  async function run(session, name, values, dims) {
    const input = new ort.Tensor('float32', values, dims);
    let outputs;
    try {
      outputs = await session.run({ [name]: input });
      if (closed) throw new Error('Pipeline closed');
      return Object.fromEntries(Object.entries(outputs).map(([key, tensor]) => {
        if (tensor.type !== 'float32' || !(tensor.data instanceof Float32Array) || !tensor.data.every(Number.isFinite)) {
          throw new Error('Invalid model output');
        }
        return [key, { type: tensor.type, dims: [...tensor.dims], data: tensor.data.slice() }];
      }));
    } finally {
      for (const tensor of Object.values(outputs ?? {})) tensor.dispose();
      input.dispose();
      values.fill(0);
    }
  }

  async function infer(frame, now = performance.now()) {
    if (closed) throw new Error('Pipeline closed');
    if (busy) throw new Error('Pipeline busy');
    validateFrame(frame);
    if (!Number.isFinite(now) || now < 0) throw new Error('Invalid frame time');
    busy = true;
    const started = performance.now();
    try {
      const detector = prepareDetector(frame);
      const detected = await run(sessions[0], 'images', detector.tensor, [1, 3, 640, 640]);
      const faces = decodeFaces(detected.output0, frame.width, frame.height).map(box => ({
        ...box, expression: unknown('skipped'), geometry: emptyGeometry('skipped'),
      }));
      const result = { width: frame.width, height: frame.height, faces,
        expressionStatus: faces.length ? 'skipped' : 'no_faces',
        landmarkStatus: faces.length ? 'skipped' : 'no_faces', modelVersion: E1_CHECKPOINT };
      if (faces.length && now - lastExpression >= 200) {
        lastExpression = now;
        result.expressionStatus = 'sampled';
        const eligible = [];
        for (const face of faces) {
          if (eligible.length >= 4) { face.expression = unknown('face_budget'); continue; }
          const prepared = prepareExpression(frame, face);
          if (prepared.status !== 'ready') face.expression = unknown(prepared.status);
          else eligible.push({ face, tensor: prepared.tensor });
        }
        if (eligible.length) {
          const batch = new Float32Array(eligible.length * 48 * 48);
          eligible.forEach((item, i) => { batch.set(item.tensor, i * 48 * 48); item.tensor.fill(0); });
          try {
            const output = (await run(sessions[1], 'faces', batch, [eligible.length, 1, 48, 48])).logits;
            if (!output || !shapeIs(output.dims, [eligible.length, 7]) || output.data.length !== eligible.length * 7) throw new Error('Invalid E1 output');
            eligible.forEach((item, i) => { item.face.expression = classifyExpression(output.data.subarray(i * 7, i * 7 + 7)); });
          } catch {
            if (closed) throw new Error('Pipeline closed');
            result.expressionStatus = 'error';
            eligible.forEach(item => { item.face.expression = unknown('model_error'); });
          }
        }
      }
      if (faces.length && now - lastMesh >= 500) {
        lastMesh = now;
        result.landmarkStatus = 'sampled';
        let attempted = 0;
        for (const face of faces) {
          if (attempted >= 2) { face.geometry = emptyGeometry('face_budget'); continue; }
          const prepared = prepareMesh(frame, face);
          if (prepared.status !== 'ready') { face.geometry = emptyGeometry(prepared.status); continue; }
          attempted++;
          try {
            const output = await run(sessions[2], 'image', prepared.tensor, [1, 3, 192, 192]);
            if (!shapeIs(output.scores?.dims, [1]) || output.scores.data.length !== 1 ||
                !shapeIs(output.landmarks?.dims, [1, 468, 3]) || output.landmarks.data.length !== 1404) throw new Error('Invalid mesh output');
            face.geometry = evaluateMesh(output.scores.data[0], output.landmarks.data);
          } catch {
            if (closed) throw new Error('Pipeline closed');
            face.geometry = emptyGeometry('engine_error');
          }
        }
      }
      if (closed) throw new Error('Pipeline closed');
      return { ...result, inferenceMs: performance.now() - started };
    } finally { busy = false; }
  }
  return { infer, dispose };
}
