// SPDX-License-Identifier: AGPL-3.0-only
import { validateFrame } from './pixels.js';
import { E1_CHECKPOINT, LABELS } from './expression.js';

const fail = code => Object.assign(new Error(code), { code });
const finiteProbability = value => Number.isFinite(value) && value >= 0 && value <= 1;
function validResult(result, frame) {
  return result && result.width === frame.width && result.height === frame.height &&
    Number.isFinite(result.inferenceMs) && result.inferenceMs >= 0 && result.modelVersion === E1_CHECKPOINT &&
    ['sampled', 'skipped', 'no_faces', 'error'].includes(result.expressionStatus) &&
    ['sampled', 'skipped', 'no_faces'].includes(result.landmarkStatus) &&
    Array.isArray(result.faces) && result.faces.length <= 20 && result.faces.every(face => {
      if (!face || ![face.x1, face.y1, face.x2, face.y2].every(Number.isFinite) ||
          face.x1 < 0 || face.y1 < 0 || face.x2 > frame.width || face.y2 > frame.height ||
          face.x2 <= face.x1 || face.y2 <= face.y1 || !finiteProbability(face.confidence)) return false;
      const expression = face.expression;
      const geometry = face.geometry;
      return expression && (expression.label === 'Unknown' ? expression.confidence === null
        : LABELS.includes(expression.label) && finiteProbability(expression.confidence)) &&
        (!expression.probabilities || Array.isArray(expression.probabilities) && expression.probabilities.length === 7 &&
          expression.probabilities.every(finiteProbability) && Math.abs(expression.probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-5) &&
        geometry && ['valid', 'skipped', 'face_budget', 'engine_error', 'invalid_roi', 'too_small', 'clipped_roi',
          'invalid_output', 'low_presence', 'clipped_mesh', 'implausible_scale', 'off_axis'].includes(geometry.status) &&
        (geometry.status === 'valid' ? Array.isArray(geometry.values) && geometry.values.length === 12 && geometry.values.every(Number.isFinite)
          : geometry.values === null) && (geometry.presence === undefined || finiteProbability(geometry.presence));
    });
}

export function createClient({
  baseUrl = new URL('.', globalThis.location.href).href,
  createWorker = () => new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }),
  loadTimeoutMs = 60000,
  frameTimeoutMs = 3000,
} = {}) {
  if (![loadTimeoutMs, frameTimeoutMs].every(value => Number.isSafeInteger(value) && value > 0 && value <= 60000)) throw new RangeError('Invalid worker deadline');
  let worker;
  let pending;
  let ready = false;
  let serial = 0;

  function stop(reason = fail('stopped')) {
    const active = worker;
    worker = undefined;
    ready = false;
    active?.terminate();
    pending?.cancel(reason);
  }

  function request(type, payload, signal = new AbortController().signal, progress) {
    signal.throwIfAborted();
    if (pending) return Promise.reject(fail('Worker busy'));
    const active = worker;
    if (!active) return Promise.reject(fail('Models not ready'));
    const id = ++serial;
    return new Promise((resolve, reject) => {
      let timer;
      const clean = () => {
        clearTimeout(timer);
        signal.removeEventListener('abort', abort);
        active.removeEventListener('message', receive);
        active.removeEventListener('error', workerError);
        active.removeEventListener('messageerror', workerError);
        if (pending?.id === id) pending = undefined;
      };
      const finish = (error, result) => {
        clean();
        if (error) {
          if (worker === active) { worker = undefined; ready = false; active.terminate(); }
          reject(error);
        } else resolve(result);
      };
      function abort() { finish(signal.reason ?? fail('stopped')); }
      function workerError(event) { event.preventDefault(); finish(fail('worker_error')); }
      function receive(event) {
        if (worker !== active || event.data?.id !== id) return;
        const reply = event.data;
        if (type === 'load' && reply.type === 'progress') {
          try { progress?.(reply.progress); } catch { finish(fail('progress_error')); }
          return;
        }
        if (reply.type === 'error') { finish(fail(typeof reply.code === 'string' ? reply.code : 'worker_error')); return; }
        if (type === 'load' && reply.type === 'ready') finish(null);
        else if (type === 'infer' && reply.type === 'result' && validResult(reply.result, payload.frame)) finish(null, reply.result);
        else finish(fail('invalid_response'));
      }
      pending = { id, cancel: error => finish(error) };
      signal.addEventListener('abort', abort, { once: true });
      active.addEventListener('message', receive);
      active.addEventListener('error', workerError);
      active.addEventListener('messageerror', workerError);
      timer = setTimeout(() => finish(fail(type === 'load' ? 'load_timeout' : 'inference_timeout')),
        type === 'load' ? loadTimeoutMs : frameTimeoutMs);
      try {
        active.postMessage({ id, type, ...payload }, type === 'infer' ? [payload.frame.pixels.buffer] : []);
      } catch { finish(fail('worker_error')); }
    });
  }

  return {
    async start({ signal, onProgress } = {}) {
      if (ready) return;
      if (worker || pending) throw fail('Worker busy');
      signal?.throwIfAborted();
      const active = createWorker();
      worker = active;
      try { await request('load', { baseUrl }, signal, onProgress); }
      catch (error) { if (worker === active) stop(error); throw error; }
      if (worker !== active) throw fail('stopped');
      ready = true;
    },
    async infer(frame, signal) {
      if (!ready) throw fail('Models not ready');
      validateFrame(frame);
      if (!(frame.pixels.buffer instanceof ArrayBuffer)) throw fail('Invalid frame buffer');
      const pixels = frame.pixels.byteOffset === 0 && frame.pixels.byteLength === frame.pixels.buffer.byteLength
        ? frame.pixels : new Uint8ClampedArray(frame.pixels);
      return request('infer', { frame: { width: frame.width, height: frame.height, pixels } }, signal);
    },
    stop,
  };
}
