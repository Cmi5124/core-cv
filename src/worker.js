// SPDX-License-Identifier: AGPL-3.0-only
import * as ort from 'onnxruntime-web/wasm';
import { loadModels } from './models.js';
import { createPipeline } from './pipeline.js';

let pipeline;
let busy = false;
globalThis.addEventListener('message', async ({ data: message }) => {
  if (!message || !Number.isSafeInteger(message.id) || message.id < 1 || !['load', 'infer'].includes(message.type)) return;
  const reply = value => globalThis.postMessage({ id: message.id, ...value });
  if (busy) { reply({ type: 'error', code: 'busy' }); return; }
  busy = true;
  try {
    if (message.type === 'load') {
      if (pipeline) throw new Error('Already initialized');
      const base = new URL(message.baseUrl);
      if (base.origin !== globalThis.location.origin) throw new Error('Invalid asset origin');
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = {
        mjs: new URL('ort/1.31.1/ort-wasm-simd-threaded.mjs', base),
        wasm: new URL('ort/1.31.1/ort-wasm-simd-threaded.wasm', base),
      };
      const bytes = await loadModels(base.href, new AbortController().signal,
        progress => reply({ type: 'progress', progress }));
      try { pipeline = await createPipeline(ort, bytes); }
      finally { for (const value of Object.values(bytes)) value.fill(0); }
      reply({ type: 'ready' });
    } else {
      if (!pipeline) throw new Error('Models not ready');
      reply({ type: 'result', result: await pipeline.infer(message.frame) });
    }
  } catch (error) {
    reply({ type: 'error', code: error?.code ?? (message.type === 'load' ? 'model_unavailable' : 'inference_error') });
  } finally {
    message.frame?.pixels?.fill(0);
    busy = false;
  }
});
