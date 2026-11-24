// SPDX-License-Identifier: AGPL-3.0-only
import { renderResults } from '../src/presentation.js';
const output = document.querySelector('#results');
const check = (value, message) => { if (!value) throw new Error(message); };
document.querySelector('#run').addEventListener('click', () => {
  try {
    const geometry = { status: 'valid', values: Array(12).fill(.2) };
    const face = { x1: 10, y1: 10, x2: 60, y2: 60, expression: { label: 'Unknown', reason: 'low_confidence' }, geometry };
    const result = { width: 100, height: 100, inferenceMs: 10, faces: [face] };
    renderResults(result);
    const details = document.querySelector('details'); details.open = true;
    const summary = details.querySelector('summary'); summary.focus();
    renderResults(result);
    check(document.querySelector('details') === details && details.open, 'disclosure replaced/closed on next frame');
    check(document.activeElement === summary, 'keyboard focus lost');
    check(details.querySelectorAll('dd').length === 12, 'missing geometry measurements');
    renderResults({ ...result, faces: [{ ...face, geometry: { status: 'skipped', values: null } }] });
    check(details.open && details.querySelectorAll('dd').length === 0, 'stale values retained or disclosure closed on skipped sample');
    renderResults({ ...result, faces: [] });
    check(document.querySelector('#faces').children.length === 0, 'stale face retained');
    output.textContent = 'ALL UI CHECKS PASSED: disclosure/focus survive updates; skipped/no-face clears measurements.';
  } catch (error) { output.textContent = 'FAIL ' + error.message; }
});
