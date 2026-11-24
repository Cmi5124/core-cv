// SPDX-License-Identifier: AGPL-3.0-only
import './style.css';
import { createClient } from './client.js';
import { createSession } from './session.js';
import { openCamera, captureFrame } from './camera.js';
import { renderResults, errorMessage } from './presentation.js';

const element = id => document.getElementById(id);
const video = element('camera');
const overlay = element('overlay');
const capture = document.createElement('canvas');
function clearResults() {
  overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
  element('faces').replaceChildren();
  element('summary').textContent = 'Không có kết quả được lưu lại.';
}
const session = createSession({ client: createClient(),
  openCamera: signal => openCamera(video, signal, () => session.stop()),
  readFrame: () => captureFrame(video, capture),
  onState({ state, code, stage, progress }) {
    const running = state === 'running';
    const pending = state === 'loading' || state === 'camera';
    element('start').disabled = running || pending;
    element('stop').disabled = !running && !pending;
    element('placeholder').hidden = running;
    element('indicator').textContent = running ? 'Đang xử lý cục bộ' : pending ? 'Đang chuẩn bị' : 'Camera tắt';
    element('indicator').classList.toggle('active', running);
    element('status').textContent = state === 'loading' ? 'Đang tải và kiểm tra ba mô hình…' + (progress?.loaded ? ` ${Math.round(progress.loaded / 1024 / 1024 * 10) / 10} MB` : '')
      : state === 'camera' ? 'Đang chờ quyền và mở camera…' : running ? 'Cả ba mô hình đã sẵn sàng. Khung hình được xử lý ngay trên thiết bị.'
      : state === 'error' ? errorMessage(code, stage) : 'Đã dừng. Camera tắt, kết quả đã được xóa.';
    if (!running) clearResults();
  },
  onResult: renderResults,
});
element('start').addEventListener('click', () => session.start());
element('stop').addEventListener('click', () => session.stop());
document.addEventListener('visibilitychange', () => { if (document.hidden) session.stop(); });
window.addEventListener('pagehide', () => session.stop());
