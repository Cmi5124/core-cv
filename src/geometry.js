// SPDX-License-Identifier: AGPL-3.0-only
export const GEOMETRY_NAMES = Object.freeze([
  'eye_open_mean', 'eye_open_asymmetry', 'brow_height_mean', 'brow_height_asymmetry',
  'mouth_open', 'mouth_width', 'mouth_corner_lift', 'mouth_corner_asymmetry',
  'nose_width', 'cheek_width', 'jaw_height', 'axis_deviation',
]);
const USED = [33, 133, 159, 145, 362, 263, 386, 374, 105, 334, 61, 291, 13, 14, 1, 98, 327, 152, 234, 454];
const f = Math.fround;
const rejected = status => ({ status, values: null });

export function measureGeometry(points) {
  if (!(points instanceof Float32Array) || points.length !== 468 * 3 || !points.every(Number.isFinite)) return rejected('invalid_output');
  const xy = index => [points[index * 3], points[index * 3 + 1]];
  if (USED.some(index => xy(index).some(value => value < f(.005) || value > f(.995)))) return rejected('clipped_mesh');
  const midpoint = (a, b) => [f(f(a[0] + b[0]) / 2), f(f(a[1] + b[1]) / 2)];
  const subtract = (a, b) => [f(a[0] - b[0]), f(a[1] - b[1])];
  const dot = (a, b) => f(f(a[0] * b[0]) + f(a[1] * b[1]));
  const centers = [midpoint(xy(33), xy(133)), midpoint(xy(362), xy(263))].sort((a, b) => a[0] - b[0]);
  const separation = subtract(centers[1], centers[0]);
  const scale = f(Math.sqrt(dot(separation, separation)));
  if (!(scale >= .12 && scale <= .76)) return rejected('implausible_scale');
  const horizontal = separation.map(value => f(value / scale));
  if (Math.abs(horizontal[1]) > .574) return rejected('off_axis');
  const vertical = [-horizontal[1], horizontal[0]];
  const center = midpoint(centers[0], centers[1]);
  const axis = Math.abs(f(dot(subtract(xy(1), center), horizontal) / scale));
  if (axis > .38) return rejected('off_axis');
  const height = (a, b) => f(dot(subtract(a, b), vertical) / scale);
  const width = (a, b) => Math.abs(f(dot(subtract(a, b), horizontal) / scale));
  const eyeA = Math.abs(height(xy(145), xy(159)));
  const eyeB = Math.abs(height(xy(374), xy(386)));
  const browA = height(midpoint(xy(33), xy(133)), xy(105));
  const browB = height(midpoint(xy(362), xy(263)), xy(334));
  const mouthCenter = midpoint(xy(13), xy(14));
  const cornerA = height(mouthCenter, xy(61));
  const cornerB = height(mouthCenter, xy(291));
  return { status: 'valid', values: [
    (eyeA + eyeB) / 2, Math.abs(eyeA - eyeB), (browA + browB) / 2, Math.abs(browA - browB),
    Math.abs(height(xy(14), xy(13))), width(xy(61), xy(291)),
    (cornerA + cornerB) / 2, Math.abs(cornerA - cornerB),
    width(xy(98), xy(327)), width(xy(234), xy(454)), height(xy(152), mouthCenter), axis,
  ] };
}

export function evaluateMesh(presence, points) {
  if (typeof presence !== 'number' || !Number.isFinite(presence) || presence < 0 || presence > 1) return rejected('invalid_output');
  if (presence < .5) return { ...rejected('low_presence'), presence };
  return { ...measureGeometry(points), presence };
}
