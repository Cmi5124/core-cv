// SPDX-License-Identifier: AGPL-3.0-only
function bounded(promise, signal, cleanup = () => {}, timeout = 20000) {
  return new Promise((resolve, reject) => {
    let finished = false;
    const timer = setTimeout(() => finish(new Error('camera_timeout')), timeout);
    const abort = () => finish(signal.reason ?? new DOMException('Stopped', 'AbortError'));
    function finish(error, value) {
      if (finished) return;
      finished = true; clearTimeout(timer); signal.removeEventListener('abort', abort);
      if (error) reject(error); else resolve(value);
    }
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
    promise.then(value => { if (finished) cleanup(value); else finish(null, value); }, error => finish(error));
  });
}

export async function openCamera(video, signal, onEnded) {
  if (!globalThis.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('camera_unavailable');
  signal.throwIfAborted();
  const stopTracks = stream => stream.getTracks().forEach(track => track.stop());
  let stream;
  let stopped = false;
  const ended = () => { if (!stopped) onEnded(); };
  function stop() {
    if (stopped) return;
    stopped = true;
    stream?.getTracks().forEach(track => track.removeEventListener('ended', ended));
    if (stream) stopTracks(stream);
    if (video.srcObject === stream) { video.pause(); video.srcObject = null; }
  }
  try {
    try {
      stream = await bounded(navigator.mediaDevices.getUserMedia({ audio: false,
        video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 } } }), signal, stopTracks);
    } catch (error) {
      if (error.name !== 'OverconstrainedError' || signal.aborted) throw error;
      stream = await bounded(navigator.mediaDevices.getUserMedia({ audio: false, video: true }), signal, stopTracks);
    }
    signal.throwIfAborted();
    video.srcObject = stream;
    stream.getVideoTracks().forEach(track => track.addEventListener('ended', ended, { once: true }));
    await bounded(video.play(), signal);
    if (!video.videoWidth || !video.videoHeight) throw new Error('camera_unavailable');
    return { stop };
  } catch (error) { stop(); throw error; }
}

export function captureFrame(video, canvas) {
  if (!video.videoWidth || !video.videoHeight || video.readyState < 2) throw new Error('camera_unavailable');
  const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  context.clearRect(0, 0, canvas.width, canvas.height);
  return { width: canvas.width, height: canvas.height, pixels };
}
