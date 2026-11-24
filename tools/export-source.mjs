// SPDX-License-Identifier: AGPL-3.0-only
import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const SOURCE_FILES = Object.freeze([
  '.gitattributes', '.gitignore', 'LICENSE', 'README.md', 'NOTICE.md', 'MODEL_CARD.md',
  'package.json', 'package-lock.json', 'vite.config.js', 'index.html', 'source.html',
  'licenses/ONNX-Runtime-LICENSE.txt', 'licenses/Pillow-LICENSE.txt', 'licenses/NumPy-LICENSE.txt',
  'public/_headers', 'public/models/core-cv.json',
  'src/camera.js', 'src/client.js', 'src/expression.js', 'src/geometry.js', 'src/main.js',
  'src/models.js', 'src/pipeline.js', 'src/pixels.js', 'src/presentation.js', 'src/session.js', 'src/style.css', 'src/worker.js',
  'tools/dev-models.mjs', 'tools/export-source.mjs', 'tools/model-specs.mjs', 'tools/sync-runtime.mjs',
  'tests/browser-harness.html', 'tests/browser-harness.js', 'tests/camera.test.mjs', 'tests/client.test.mjs',
  'tests/dev-models.test.mjs', 'tests/export-source.test.mjs', 'tests/fixtures/oracle.json',
  'tests/models.test.mjs', 'tests/numerics.test.mjs', 'tests/pipeline.test.mjs', 'tests/session.test.mjs',
  'tests/static-boundary.test.mjs', 'tests/presentation.test.mjs', 'tests/ui-harness.html', 'tests/ui-harness.js',
  'reproduce/export_yolo26_face.py', 'reproduce/pack_mesh.py', 'reproduce/test_runtime.py',
  'reproduce/expression/dataset.py', 'reproduce/expression/metrics.py', 'reproduce/expression/prepare.py',
  'reproduce/expression/review9.py', 'reproduce/expression/review9-config.json',
  'reproduce/expression/runtime.py', 'reproduce/expression/train_support.py',
]);
const forbidden = /(^|\/)(?:\.git|\.env[^/]*|node_modules|dist|local-models|apps|data)(\/|$)|\.(?:onnx|pt|npz|csv|pem|key|zip)$/i;
const secrets = /(?:gh[pousr]_[a-zA-Z0-9]{30,}|github_pat_[a-zA-Z0-9_]{30,}|sk-[a-zA-Z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)/;
const privateImport = /(?:from\s+|import\s*\(?\s*)['"](?:[^'"]*apps\/(?:portal|learning-monitor)|\.\.\/\.\.\/)|(?:C:\\Users\\ADMIN|\/Users\/[^/]+\/Documents\/Project)/;

export async function exportSource({ root = fileURLToPath(new URL('../', import.meta.url)), destination, files = SOURCE_FILES }) {
  const base = await realpath(root);
  const output = path.resolve(destination);
  if (output === base || output.startsWith(base + path.sep)) throw new Error('Unsafe source destination');
  try { await lstat(output); throw new Error('Destination already exists'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const parent = await realpath(path.dirname(output));
  if (parent !== path.dirname(output)) throw new Error('Unsafe symlink destination');
  if (!Array.isArray(files) || !files.length || new Set(files).size !== files.length) throw new Error('Unsafe source list');
  const contents = [];
  for (const name of [...files].sort()) {
    if (typeof name !== 'string' || name.includes('\\') || name.includes(':') || name.startsWith('/') ||
        name.split('/').some(part => !part || part === '.' || part === '..') || forbidden.test(name) ||
        !/^(?:[a-zA-Z0-9_.-]+\/)*[a-zA-Z0-9_.-]+$/.test(name)) throw new Error('Unsafe or forbidden source path');
    let current = base;
    for (const part of name.split('/')) {
      current = path.join(current, part);
      if ((await lstat(current)).isSymbolicLink()) throw new Error('Source symlink forbidden');
    }
    const stat = await lstat(current);
    if (!stat.isFile() || stat.size > 1024 * 1024) throw new Error('Unsafe source file');
    const source = new TextDecoder('utf-8', { fatal: true }).decode(await readFile(current)).replace(/\r\n/g, '\n');
    if (source.includes('\0')) throw new Error('Unsafe binary source file');
    if (secrets.test(source)) throw new Error('Detected secret in source');
    if (privateImport.test(source)) throw new Error('Forbidden private source import');
    const bytes = Buffer.from(source, 'utf8');
    contents.push({ name, bytes });
  }
  const manifest = { schemaVersion: 1, license: 'AGPL-3.0-only', files: contents.map(({ name, bytes }) => ({
    path: name, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex'),
  })) };
  await mkdir(output);
  for (const { name, bytes } of contents) {
    await mkdir(path.dirname(path.join(output, name)), { recursive: true });
    await writeFile(path.join(output, name), bytes, { flag: 'wx' });
  }
  await writeFile(path.join(output, 'SOURCE_MANIFEST.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
  return manifest;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('Usage: node tools/export-source.mjs /absolute/new/candidate');
  const manifest = await exportSource({ destination: process.argv[2] });
  console.log(`Exported ${manifest.files.length} Core CV source files`);
}
