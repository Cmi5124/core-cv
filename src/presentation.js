// SPDX-License-Identifier: AGPL-3.0-only
import { GEOMETRY_NAMES } from "./geometry.js";
const element = id => document.getElementById(id);
const messages = { models_unpublished: 'Bản này chưa có đủ model được phát hành. Camera vẫn tắt. Bạn có thể xem mã nguồn và trạng thái model.',
  load_timeout: 'Tải mô hình quá lâu. Kiểm tra mạng rồi thử lại.', inference_timeout: 'Thiết bị xử lý quá lâu. Phiên đã dừng; bạn có thể thử lại.',
  NotAllowedError: 'Chưa được cấp quyền camera. Cho phép camera trong trình duyệt rồi thử lại.',
  NotFoundError: 'Không tìm thấy camera. Kết nối camera rồi thử lại.', camera_timeout: 'Camera chưa phản hồi. Kiểm tra quyền và thử lại.',
  NotReadableError: 'Camera đang bận hoặc chưa thể mở. Đóng ứng dụng đang dùng camera rồi thử lại.' };
const reasons = { skipped: 'Chưa đến lượt lấy mẫu', face_budget: 'Giới hạn số khuôn mặt mỗi lượt', edge_crop: 'Khuôn mặt sát mép ảnh',
  clipped_roi: 'Vùng khuôn mặt bị cắt', too_small: 'Khuôn mặt quá nhỏ', flat: 'Thiếu chi tiết', blurred: 'Ảnh mờ', low_contrast: 'Độ tương phản thấp',
  low_confidence: 'Độ tin cậy chưa đủ', low_presence: 'Mesh chưa đủ tin cậy', clipped_mesh: 'Mesh sát mép', off_axis: 'Góc mặt chưa phù hợp',
  implausible_scale: 'Tỉ lệ mesh chưa hợp lệ', engine_error: 'Mô hình gặp lỗi', model_error: 'Mô hình gặp lỗi' };
function text(tag, value, className) { const node = document.createElement(tag); node.textContent = value; if (className) node.className = className; return node; }

export const errorMessage = (code, stage) => messages[code] ?? (stage === 'model'
  ? 'Không thể tải hoặc xác minh mô hình. Phiên chưa mở camera; kiểm tra mạng hoặc bản phát hành rồi thử lại.'
  : stage === 'camera' ? 'Không thể mở camera trên thiết bị này. Kiểm tra quyền và camera rồi thử lại.'
  : 'Mô hình gặp lỗi khi xử lý. Phiên đã dừng; bạn có thể thử lại.');
export function renderResults(result) {
    const overlay = element('overlay');
    overlay.width = result.width; overlay.height = result.height;
    const context = overlay.getContext('2d');
    context.strokeStyle = '#7da8ff'; context.lineWidth = 2;
    element('summary').textContent = `${result.faces.length} khuôn mặt · ${Math.round(result.inferenceMs)} ms/lượt · kết quả cục bộ`;
    const container = element('faces');
    while (container.children.length > result.faces.length) container.lastElementChild.remove();
    result.faces.forEach((face, i) => {
      context.strokeRect(face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1);
      context.fillStyle = '#7da8ff'; context.font = '16px sans-serif'; context.fillText(String(i + 1), face.x1 + 4, face.y1 + 18);
      let card = container.children[i];
      if (!card) {
        card = text('article', '', 'face-card');
        const details = document.createElement('details');
        details.append(text('summary', '12 số đo hình học'), text('p', ''), document.createElement('dl'));
        card.append(text('h3', `Khuôn mặt ${i + 1}`), text('p', ''), details);
        container.append(card);
      }
      const expression = face.expression;
      card.children[1].textContent = expression.label === 'Unknown' ? `CNN: Unknown · ${reasons[expression.reason] ?? 'Chưa đủ dữ liệu'}` : `CNN: ${expression.label} · ${(expression.confidence * 100).toFixed(1)}%`;
      const details = card.querySelector('details');
      const list = details.querySelector('dl');
      list.replaceChildren();
      if (face.geometry.status === 'valid') {
        details.querySelector('p').textContent = 'Số đo của lượt hiện tại; số thứ tự không định danh một người.';
        GEOMETRY_NAMES.forEach((name, j) => list.append(text('dt', name), text('dd', face.geometry.values[j].toFixed(4))));
      } else details.querySelector('p').textContent = 'Mesh: ' + (reasons[face.geometry.status] ?? 'Chưa đủ dữ liệu');
    });
}
