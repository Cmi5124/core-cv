// SPDX-License-Identifier: AGPL-3.0-only
import { MODEL_PINS } from '../tools/model-specs.mjs';
const MiB = 1024 * 1024;
export const CONTRACTS = Object.freeze({ faceDetector: 'yolo26-face-v1', expression: 'e1-gray48-v1', landmarks: 'facemesh468-v1' });
const LIMITS = { faceDetector: 16 * MiB, expression: 8 * MiB, landmarks: 4 * MiB };

export class ModelError extends Error {
  constructor(code) { super(code); this.name = 'ModelError'; this.code = code; }
}

async function readAsset(url, limit, signal, cache, progress) {
  signal.throwIfAborted();
  const response = await fetch(url, { signal, cache, credentials: 'omit', redirect: 'error' });
  if (!response.ok || !response.body || Number(response.headers.get('Content-Length')) > limit) {
    await response.body?.cancel();
    throw new ModelError('model_unavailable');
  }
  const reader = response.body.getReader();
  const cancel = () => { void reader.cancel(signal.reason).catch(() => {}); };
  signal.addEventListener('abort', cancel, { once: true });
  try {
    if (signal.aborted) { cancel(); signal.throwIfAborted(); }
    const output = new Uint8Array(limit);
    let count = 0;
    for (;;) {
      const { done, value } = await reader.read();
      signal.throwIfAborted();
      if (done) return output.subarray(0, count);
      if (value.length > limit - count) {
        await reader.cancel();
        throw new ModelError('invalid_model_size');
      }
      output.set(value, count);
      count += value.length;
      progress?.(count, limit);
    }
  } finally {
    signal.removeEventListener('abort', cancel);
    reader.releaseLock();
  }
}

export async function loadModels(baseUrl, signal, onProgress, profile = MODEL_PINS) {
  const base = new URL(baseUrl);
  if (base.origin !== globalThis.location?.origin || base.username || base.password) throw new ModelError('invalid_model_origin');
  const manifestUrl = new URL('models/core-cv.json', base);
  const raw = await readAsset(manifestUrl, 8192, signal, 'no-store');
  let manifest;
  try { manifest = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(raw)); }
  catch { throw new ModelError('invalid_manifest'); }
  if (manifest?.schemaVersion !== 1 || !manifest.models || typeof manifest.models !== 'object' || Array.isArray(manifest.models)) {
    throw new ModelError('invalid_manifest');
  }
  // Validate the complete manifest before requesting any model bytes.
  for (const [name, contract] of Object.entries(CONTRACTS)) {
    const item = manifest.models[name];
    if (item === null) throw new ModelError('models_unpublished');
    if (!item || typeof item.file !== 'string' || !/^[a-z0-9][a-z0-9_-]{0,95}\.onnx$/.test(item.file) ||
        typeof item.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(item.sha256) ||
        !Number.isSafeInteger(item.bytes) || item.bytes < 1 || item.bytes > LIMITS[name] || item.contract !== contract) {
      throw new ModelError('invalid_manifest');
    }
    const expected = profile[name];
    if (!expected || item.file !== expected.file || item.sha256 !== expected.sha256 || item.bytes !== expected.bytes || item.contract !== expected.contract) {
      throw new ModelError('model_profile_mismatch');
    }
  }
  const models = {};
  try {
    for (const name of Object.keys(CONTRACTS)) {
      const item = manifest.models[name];
      const bytes = await readAsset(new URL(item.file, manifestUrl), item.bytes, signal, 'default',
        (loaded, total) => onProgress?.({ stage: name, loaded, total }));
      if (bytes.length !== item.bytes) throw new ModelError('invalid_model_size');
      const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
      signal.throwIfAborted();
      if (Array.from(hash, value => value.toString(16).padStart(2, '0')).join('') !== item.sha256) throw new ModelError('invalid_model_digest');
      models[name] = bytes;
    }
    return models;
  } catch (error) {
    for (const bytes of Object.values(models)) bytes.fill(0);
    throw error;
  }
}
