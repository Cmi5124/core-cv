// SPDX-License-Identifier: AGPL-3.0-only
import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { localModelsPlugin } from './tools/dev-models.mjs';

export default defineConfig({
  envDir: false,
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'] },
  build: { rolldownOptions: { input: { main: 'index.html', source: 'source.html' } } },
  server: { host: '127.0.0.1', port: 5186, strictPort: true,
    fs: { strict: true, allow: [fileURLToPath(new URL('.', import.meta.url))],
      deny: ['.env', '.env.*', '*.{crt,pem,key,p12,pfx,cer,der}', '.npmrc', '.yarnrc.yml', '**/.git/**', '**/*.onnx', '**/*.pt', '**/local-models/**'] } },
  plugins: [localModelsPlugin({ directory: process.env.CORE_CV_LOCAL_MODEL_DIR || fileURLToPath(new URL('./local-models/', import.meta.url)) })],
  worker: { format: 'es' },
});
