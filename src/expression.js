// SPDX-License-Identifier: AGPL-3.0-only
export const LABELS = Object.freeze(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']);
export const E1_POLICY = Object.freeze({ temperature: 1.231144413344916, minProbability: .63, minMargin: .05 });
export const E1_CHECKPOINT = 'c546b469b4ab815fc83b28a2c7b695cb8b9bae20deb846ebf2aa16ea914752cc';

export function classifyExpression(logits) {
  if (!logits || logits.length !== 7 || !Array.from(logits).every(value => typeof value === 'number' && Number.isFinite(value))) {
    throw new Error('Invalid E1 expression logits');
  }
  const scaled = Array.from(logits, value => value / E1_POLICY.temperature);
  const maximum = Math.max(...scaled);
  const weights = scaled.map(value => Math.exp(value - maximum));
  const sum = weights.reduce((total, value) => total + value, 0);
  const probabilities = weights.map(value => value / sum);
  const sorted = probabilities.map((probability, index) => ({ probability, index })).sort((a, b) => b.probability - a.probability);
  const accepted = sorted[0].probability >= E1_POLICY.minProbability && sorted[0].probability - sorted[1].probability >= E1_POLICY.minMargin;
  return {
    label: accepted ? LABELS[sorted[0].index] : 'Unknown',
    confidence: accepted ? sorted[0].probability : null,
    reason: accepted ? null : 'low_confidence', probabilities,
  };
}
