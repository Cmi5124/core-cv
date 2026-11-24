import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import test from 'node:test';

async function fixture(t) {
  const { exportSource } = await import('../tools/export-source.mjs');
  const folder = await mkdtemp(path.join(tmpdir(), 'core-cv-export-test-'));
  const root = path.join(folder, 'source'); await mkdir(root);
  await mkdir(path.join(root, 'src'));
  await writeFile(path.join(root, 'src/main.js'), 'export const value = 1;\n');
  await writeFile(path.join(folder, 'private.js'), 'private application');
  await writeFile(path.join(root, '.env'), 'SECRET=private');
  t.after(async () => { assert.ok(folder.startsWith(path.join(tmpdir(), 'core-cv-export-test-'))); await rm(folder, { recursive: true }); });
  return { exportSource, folder, root, destination: path.join(folder, 'candidate') };
}
test('source export copies only explicit safe paths and records file digests', async t => {
  const f = await fixture(t);
  const record = await f.exportSource({ ...f, files: ['src/main.js'] });
  assert.deepEqual(record.files.map(file => file.path), ['src/main.js']);
  assert.match(record.files[0].sha256, /^[a-f0-9]{64}$/);
  assert.equal(await readFile(path.join(f.destination, 'src/main.js'), 'utf8'), 'export const value = 1;\n');
  await assert.rejects(readFile(path.join(f.destination, '.env')), /ENOENT/);
  await assert.rejects(f.exportSource({ ...f, files: ['src/main.js'] }), /exist/i);
});
test('source export refuses traversal, private paths, secrets, weights and symlink escape', async t => {
  const f = await fixture(t);
  for (const filename of ['../private.js', '.env', 'models/model.onnx', 'C:/secret.js', 'src/../../private.js', 'apps/portal/main.js']) {
    await assert.rejects(f.exportSource({ ...f, files: [filename] }), /unsafe|forbidden/i);
  }
  await symlink(f.folder, path.join(f.root, 'linked'), 'junction');
  await assert.rejects(f.exportSource({ ...f, files: ['linked/private.js'] }), /symlink/i);
  await writeFile(path.join(f.root, 'src/token.js'), "const token = '" + 'ghp_' + '1'.repeat(36) + "';\n");
  await assert.rejects(f.exportSource({ ...f, files: ['src/token.js'] }), /secret/i);
});
