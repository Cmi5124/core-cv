// SPDX-License-Identifier: AGPL-3.0-only
import { mkdir, copyFile, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const source = path.dirname(fileURLToPath(import.meta.resolve('onnxruntime-web/wasm')));
const packageRoot = path.dirname(source);
const metadata = JSON.parse(await readFile(path.join(packageRoot, 'package.json'), 'utf8'));
if (metadata.version !== '1.31.1') throw new Error('Unexpected ONNX Runtime version');
const destination = new URL('../public/ort/1.31.1/', import.meta.url);
await mkdir(destination, { recursive: true });
for (const file of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await copyFile(path.join(source, file), new URL(file, destination));
}
for (const file of ['ONNX-Runtime-LICENSE.txt']) {
  await copyFile(new URL('../licenses/' + file, import.meta.url), new URL(file, destination));
}
await copyFile(new URL('../LICENSE', import.meta.url), new URL('../public/LICENSE', import.meta.url));
