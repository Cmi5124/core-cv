# SPDX-License-Identifier: AGPL-3.0-only
# Extracted/adapted for standalone Core CV, 2026-11-24. No product service code.
"""Classification and selective prediction metrics for a fixed policy."""
from dataclasses import asdict

import numpy as np

from runtime import ExpressionPolicy, LABELS, accepted, probabilities


def scores(probability, labels):
    if not len(labels):
        return None
    pred = probability.argmax(axis=1)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(confusion, (labels, pred), 1)
    tp = confusion.diagonal()
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = tp / np.maximum(predicted, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # Accepted subsets may contain fewer than seven true classes.
    present = support > 0
    confidence = probability.max(axis=1)
    correct = pred == labels
    buckets = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(
        float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        for bucket in np.unique(buckets) if (mask := buckets == bucket).any()
    )
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                            "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(LABELS)},
        "confusion_matrix": confusion.tolist(),
        "ece_15": float(ece),
        "nll": float(-np.log(probability[np.arange(len(labels)), labels].clip(1e-15)).mean()),
        "brier": float(((probability - np.eye(len(LABELS))[labels]) ** 2).sum(axis=1).mean()),
        "false_confident_count_at_0_8": int(((confidence >= 0.8) & ~correct).sum()),
    }


def calibrate(logits, labels, quality_ok, config):
    def nll(temperature):
        p = probabilities(logits, temperature)
        return -np.log(p[np.arange(len(labels)), labels].clip(1e-15)).mean()

    temperature = float(min(np.geomspace(0.25, 8, 81), key=nll))
    probability = probabilities(logits, temperature)
    correct = probability.argmax(axis=1) == labels
    best = ExpressionPolicy(temperature, None, 0)
    best_count = 0
    for threshold in np.arange(0.35, 0.951, 0.05):
        for margin in (0.05, 0.10, 0.20):
            policy = ExpressionPolicy(temperature, round(float(threshold), 2), margin)
            accept = accepted(probability, policy) & quality_ok
            count = int(accept.sum())
            if (count > best_count and count >= config["minimum_validation_accepted"]
                    and count / len(labels) >= config["minimum_validation_coverage"]
                    and correct[accept].mean() >= config["minimum_validation_accuracy"]):
                best, best_count = policy, count
    return best


def evaluate(logits, labels, quality_ok, policy):
    probability = probabilities(logits, policy.temperature)
    keep = accepted(probability, policy) & quality_ok
    return {
        "policy": asdict(policy),
        "before_temperature": scores(probabilities(logits, 1), labels),
        "without_gating": scores(probability, labels),
        "with_gating": scores(probability[keep], labels[keep]),
        "coverage": float(keep.mean()), "rejection_rate": float(1 - keep.mean()),
        "accepted_count": int(keep.sum()), "input_count": len(labels),
    }
