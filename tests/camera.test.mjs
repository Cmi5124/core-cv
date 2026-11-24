import assert from 'node:assert/strict';
import test from 'node:test';

async function fixture(t, getUserMedia) {
  const descriptors = ['navigator', 'isSecureContext'].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]);
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { mediaDevices: { getUserMedia } } });
  Object.defineProperty(globalThis, 'isSecureContext', { configurable: true, value: true });
  t.after(() => descriptors.forEach(([key, descriptor]) => descriptor ? Object.defineProperty(globalThis, key, descriptor) : delete globalThis[key]));
  const { openCamera } = await import('../src/camera.js');
  const track = new EventTarget(); track.stops = 0; track.stop = () => track.stops++;
  const stream = { getTracks: () => [track], getVideoTracks: () => [track] };
  const video = { srcObject: null, videoWidth: 640, videoHeight: 480, async play() {}, pause() {} };
  return { openCamera, track, stream, video };
}
test('automatic camera selection requests no audio and falls back only for constraints', async t => {
  const requests = []; let stream;
  const f = await fixture(t, async request => { requests.push(request); if (requests.length === 1) throw new DOMException('Constraints', 'OverconstrainedError'); return stream; });
  stream = f.stream;
  const camera = await f.openCamera(f.video, new AbortController().signal, () => {});
  assert.equal(requests[0].audio, false); assert.equal(requests[0].video.facingMode.ideal, 'user');
  assert.deepEqual(requests[1], { audio: false, video: true });
  camera.stop(); camera.stop();
  assert.equal(f.track.stops, 1); assert.equal(f.video.srcObject, null);
});
test('permission denial is not retried as a different camera request', async t => {
  let count = 0;
  const f = await fixture(t, async () => { count++; throw new DOMException('Denied', 'NotAllowedError'); });
  await assert.rejects(f.openCamera(f.video, new AbortController().signal, () => {}), error => error.name === 'NotAllowedError');
  assert.equal(count, 1);
});
test('permission granted after abort releases the newly obtained stream', async t => {
  let accept;
  const f = await fixture(t, () => new Promise(resolve => { accept = resolve; }));
  const controller = new AbortController();
  const opening = f.openCamera(f.video, controller.signal, () => {});
  controller.abort();
  await assert.rejects(opening, error => error.name === 'AbortError');
  accept(f.stream); await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.track.stops, 1); assert.equal(f.video.srcObject, null);
});
test('play failure releases camera tracks and stopped tracks do not trigger another session', async t => {
  let stream; let ended = 0;
  const f = await fixture(t, async () => stream); stream = f.stream;
  f.video.play = async () => { throw new Error('Playback failed'); };
  await assert.rejects(f.openCamera(f.video, new AbortController().signal, () => ended++), /Playback failed/);
  f.track.dispatchEvent(new Event('ended'));
  assert.equal(f.track.stops, 1); assert.equal(f.video.srcObject, null); assert.equal(ended, 0);
});
