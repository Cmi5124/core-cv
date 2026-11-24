// SPDX-License-Identifier: AGPL-3.0-only
// Fixed-point bilinear resampling is compatible with Pillow's 8-bit image path.
const PRECISION = 2 ** 22;

export function validateFrame(frame) {
  if (!frame || !Number.isSafeInteger(frame.width) || !Number.isSafeInteger(frame.height) ||
      frame.width < 1 || frame.height < 1 || frame.width > 640 || frame.height > 640 ||
      !(frame.pixels instanceof Uint8Array || frame.pixels instanceof Uint8ClampedArray) ||
      frame.pixels.length !== frame.width * frame.height * 4) throw new RangeError('Invalid RGBA frame');
}

function weights(input, output) {
  const ratio = input / output;
  const support = Math.max(1, ratio);
  return Array.from({ length: output }, (_, i) => {
    const center = (i + .5) * ratio;
    const start = Math.max(0, Math.trunc(center - support + .5));
    const end = Math.min(input, Math.trunc(center + support + .5));
    const values = Array.from({ length: end - start }, (_, offset) =>
      Math.max(0, 1 - Math.abs((start + offset - center + .5) / support)));
    const total = values.reduce((sum, value) => sum + value, 0);
    return { start, values: values.map(value => Math.floor(value / total * PRECISION + .5)) };
  });
}

export function resizeBilinear(pixels, width, height, targetWidth, targetHeight, channels) {
  if (!(pixels instanceof Uint8Array) || pixels.length !== width * height * channels ||
      ![width, height, targetWidth, targetHeight].every(value => Number.isSafeInteger(value) && value >= 1 && value <= 640) ||
      ![1, 3].includes(channels)) throw new RangeError('Invalid resize dimensions');
  let horizontal = pixels;
  if (targetWidth !== width) {
    horizontal = new Uint8Array(targetWidth * height * channels);
    const columns = weights(width, targetWidth);
    for (let y = 0; y < height; y++) for (let x = 0; x < targetWidth; x++) {
      const { start, values } = columns[x];
      for (let channel = 0; channel < channels; channel++) {
        let sum = PRECISION / 2;
        for (let i = 0; i < values.length; i++) sum += pixels[(y * width + start + i) * channels + channel] * values[i];
        horizontal[(y * targetWidth + x) * channels + channel] = Math.min(255, Math.max(0, Math.floor(sum / PRECISION)));
      }
    }
  }
  if (targetHeight === height) return horizontal;
  const output = new Uint8Array(targetWidth * targetHeight * channels);
  const rows = weights(height, targetHeight);
  for (let y = 0; y < targetHeight; y++) {
    const { start, values } = rows[y];
    for (let x = 0; x < targetWidth; x++) for (let channel = 0; channel < channels; channel++) {
      let sum = PRECISION / 2;
      for (let i = 0; i < values.length; i++) sum += horizontal[((start + i) * targetWidth + x) * channels + channel] * values[i];
      output[(y * targetWidth + x) * channels + channel] = Math.min(255, Math.max(0, Math.floor(sum / PRECISION)));
    }
  }
  return output;
}

function region(frame, left, top, width, height, grayscale) {
  const channels = grayscale ? 1 : 3;
  const output = new Uint8Array(width * height * channels);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const source = ((top + y) * frame.width + left + x) * 4;
    const target = (y * width + x) * channels;
    const [red, green, blue] = frame.pixels.subarray(source, source + 3);
    if (grayscale) output[target] = Math.floor((19595 * red + 38470 * green + 7471 * blue + 32768) / 65536);
    else output.set([red, green, blue], target);
  }
  return output;
}

function planar(pixels, channels) {
  const area = pixels.length / channels;
  const output = new Float32Array(pixels.length);
  for (let i = 0; i < area; i++) for (let channel = 0; channel < channels; channel++) {
    output[channel * area + i] = pixels[i * channels + channel] / 255;
  }
  return output;
}

function validBox(box) {
  return box && [box.x1, box.y1, box.x2, box.y2].every(Number.isFinite) && box.x2 > box.x1 && box.y2 > box.y1;
}

// Match NumPy's float32 reduction ordering, including its 128-element pairwise blocks.
// Reference: NumPy 2.4.6 loops_utils.h.src (BSD-3-Clause, licenses/NumPy-LICENSE.txt).
function sumFloat32(values, start = 0, count = values.length) {
  const f = Math.fround;
  if (count < 8) {
    let total = -0;
    for (let i = 0; i < count; i++) total = f(total + values[start + i]);
    return total;
  }
  if (count > 128) {
    const half = Math.floor(count / 2 / 8) * 8;
    return f(sumFloat32(values, start, half) + sumFloat32(values, start + half, count - half));
  }
  const sums = values.slice(start, start + 8);
  let i = 8;
  for (; i < count - count % 8; i += 8) for (let j = 0; j < 8; j++) sums[j] = f(sums[j] + values[start + i + j]);
  let total = f(f(f(sums[0] + sums[1]) + f(sums[2] + sums[3])) + f(f(sums[4] + sums[5]) + f(sums[6] + sums[7])));
  for (; i < count; i++) total = f(total + values[start + i]);
  return total;
}
function varianceFloat32(values) {
  const mean = Math.fround(sumFloat32(values) / values.length);
  const squares = Float32Array.from(values, value => { const delta = Math.fround(value - mean); return Math.fround(delta * delta); });
  return Math.fround(sumFloat32(squares) / squares.length);
}
export function qualityReason(pixels, width, height) {
  if (!(pixels instanceof Uint8Array || pixels instanceof Uint8ClampedArray) ||
      ![width, height].every(value => Number.isSafeInteger(value) && value >= 3 && value <= 640) || pixels.length !== width * height) throw new RangeError('Invalid quality pixels');
  if (Math.fround(Math.sqrt(varianceFloat32(Float32Array.from(pixels)))) < 9) return 'low_contrast';
  const laplacian = new Float32Array((width - 2) * (height - 2));
  let count = 0;
  for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
    const offset = y * width + x;
    const lap = pixels[offset - width] + pixels[offset + width] + pixels[offset - 1] + pixels[offset + 1] - 4 * pixels[offset];
    laplacian[count++] = lap;
  }
  return varianceFloat32(laplacian) < 18 ? 'blurred' : null;
}

export function prepareExpression(frame, box) {
  validateFrame(frame);
  if (!validBox(box)) return { status: 'invalid_crop' };
  const { x1, y1, x2, y2 } = box;
  if (x1 <= 0 || y1 <= 0 || x2 >= frame.width || y2 >= frame.height) return { status: 'edge_crop' };
  if (Math.min(x2 - x1, y2 - y1) < 40) return { status: 'too_small' };
  const left = Math.floor(x1);
  const top = Math.floor(y1);
  const width = Math.ceil(x2) - left;
  const height = Math.ceil(y2) - top;
  const pixels = resizeBilinear(region(frame, left, top, width, height, true), width, height, 48, 48, 1);
  const reason = qualityReason(pixels, 48, 48);
  return reason ? { status: reason } : { status: 'ready', pixels, tensor: planar(pixels, 1) };
}

export function prepareMesh(frame, box) {
  validateFrame(frame);
  if (!validBox(box)) return { status: 'invalid_roi' };
  if (Math.min(box.x2 - box.x1, box.y2 - box.y1) < 52) return { status: 'too_small' };
  const side = Math.ceil(1.12 * Math.max(box.x2 - box.x1, box.y2 - box.y1));
  const left = Math.floor((box.x1 + box.x2 - side) / 2);
  const top = Math.floor((box.y1 + box.y2 - side) / 2);
  if (left < 0 || top < 0 || left + side > frame.width || top + side > frame.height) return { status: 'clipped_roi' };
  const pixels = resizeBilinear(region(frame, left, top, side, side, false), side, side, 192, 192, 3);
  return { status: 'ready', pixels, tensor: planar(pixels, 3), roi: [left, top, side] };
}

function nearestEven(value) {
  const lower = Math.floor(value);
  return value - lower === .5 ? lower + lower % 2 : Math.round(value);
}

export function prepareDetector(frame) {
  validateFrame(frame);
  const scale = 640 / Math.max(frame.width, frame.height);
  const width = Math.max(1, nearestEven(frame.width * scale));
  const height = Math.max(1, nearestEven(frame.height * scale));
  const left = Math.floor((640 - width) / 2);
  const top = Math.floor((640 - height) / 2);
  const resized = resizeBilinear(region(frame, 0, 0, frame.width, frame.height, false), frame.width, frame.height, width, height, 3);
  const pixels = new Uint8Array(640 * 640 * 3).fill(114);
  for (let y = 0; y < height; y++) pixels.set(resized.subarray(y * width * 3, (y + 1) * width * 3), ((top + y) * 640 + left) * 3);
  return { tensor: planar(pixels, 3), scale, left, top };
}

export function decodeFaces(output, width, height) {
  if (!output || output.type !== 'float32' || String(output.dims) !== '1,300,6' ||
      !(output.data instanceof Float32Array) || output.data.length !== 1800 ||
      ![width, height].every(value => Number.isSafeInteger(value) && value >= 1 && value <= 640)) {
    throw new Error('Invalid YOLO26 tensor output');
  }
  const scale = 640 / Math.max(width, height);
  const padX = Math.floor((640 - nearestEven(width * scale)) / 2);
  const padY = Math.floor((640 - nearestEven(height * scale)) / 2);
  const boxes = [];
  const clip = (value, size) => Math.min(size, Math.max(0, value));
  for (let i = 0; i < 1800; i += 6) {
    const [left, top, right, bottom, confidence, classId] = output.data.subarray(i, i + 6);
    if (![left, top, right, bottom, confidence].every(Number.isFinite) || confidence < 0 || confidence > 1 || classId !== 0) {
      throw new Error('Invalid YOLO26 face row');
    }
    if (confidence < .4) continue;
    const x1 = clip((left - padX) / scale, width);
    const y1 = clip((top - padY) / scale, height);
    const x2 = clip((right - padX) / scale, width);
    const y2 = clip((bottom - padY) / scale, height);
    if (x2 > x1 && y2 > y1) boxes.push({ x1, y1, x2, y2, confidence });
  }
  return boxes.sort((a, b) => b.confidence - a.confidence).slice(0, 20);
}
