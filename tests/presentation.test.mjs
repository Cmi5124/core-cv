import assert from 'node:assert/strict';
import test from 'node:test';
import { errorMessage } from '../src/presentation.js';

test('model integrity, network and runtime failures are distinguished from camera failures', () => {
  for (const code of ['model_profile_mismatch', 'invalid_manifest', 'invalid_model_digest', 'model_unavailable', 'invalid_model_size', 'worker_error']) {
    assert.match(errorMessage(code, 'model'), /mô hình/i);
    assert.doesNotMatch(errorMessage(code, 'model'), /quyền camera/i);
  }
  assert.match(errorMessage('worker_error', 'inference'), /xử lý/i);
  assert.match(errorMessage('camera_unavailable', 'camera'), /camera/i);
});
