# SPDX-License-Identifier: AGPL-3.0-only
# Extracted/adapted for standalone Core CV, 2026-11-24. No product service code.
"""Train and select Review 9 classifiers with frozen FER2013 splits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_support import FaceDataset, collect, export_and_check, benchmark, load_prepared, sha_file, train_epoch, write_report
from runtime import MAX_BATCH, ExpressionModel, ExpressionPolicy, LABELS, SIDE, accepted, normalize_pixels, probabilities, quality_reason
from metrics import evaluate, scores

CONFIG = Path(__file__).with_name("review9-config.json")


def conv(source, target, stride, separable):
    if separable:
        return nn.Sequential(nn.Conv2d(source, source, 3, stride, 1, groups=source, bias=False),
                             nn.BatchNorm2d(source), nn.ReLU(),
                             nn.Conv2d(source, target, 1, bias=False), nn.BatchNorm2d(target))
    return nn.Sequential(nn.Conv2d(source, target, 3, stride, 1, bias=False), nn.BatchNorm2d(target))


class Residual(nn.Module):
    def __init__(self, source, target, stride, separable):
        super().__init__()
        self.entry = conv(source, target, stride, separable)
        self.block = conv(target, target, 1, separable)

    def forward(self, inputs):
        base = self.entry(inputs).relu()
        return (base + self.block(base)).relu()


def classifier(spec, dropout):
    separable = spec["backbone"] == "separable"
    return nn.Sequential(
        conv(1, 24, 1, False), nn.ReLU(), nn.MaxPool2d(2),
        Residual(24, 32, 1, separable), Residual(32, 64, 2, separable),
        Residual(64, 96, 2, separable), nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        nn.Dropout(dropout), nn.Linear(96 * 4, len(LABELS)),
    ).to(memory_format=torch.channels_last)


def setup(seed):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def calibration(logits, labels, quality, baseline, minimum):
    def nll(t):
        return -np.log(probabilities(logits, t)[np.arange(len(labels)), labels].clip(1e-15)).mean()
    temperature = float(min(np.geomspace(.25, 8, 101), key=nll))
    p = probabilities(logits, temperature)
    correct = p.argmax(1) == labels
    floor = baseline["with_gating"]["accuracy"]
    best = ExpressionPolicy(temperature, None, 0)
    size = 0
    curve = []
    for threshold in np.arange(.35, .991, .01):
        for margin in (.05, .10, .15, .20):
            policy = ExpressionPolicy(temperature, round(float(threshold), 2), margin)
            keep = accepted(p, policy) & quality
            n = int(keep.sum())
            accuracy = float(correct[keep].mean()) if n else None
            curve.append({"threshold": policy.min_probability, "margin": margin, "coverage": n / len(labels),
                          "selective_accuracy": accuracy, "accepted": n})
            if n > size and n >= minimum and accuracy >= floor:
                best, size = policy, n
    return best, curve


def detailed(logits, labels, quality, policy):
    report = evaluate(logits, labels, quality, policy)
    p = probabilities(logits, policy.temperature)
    keep = accepted(p, policy) & quality
    report["rejection_by_class"] = {}
    for index, name in enumerate(LABELS):
        mask = labels == index
        total = int(mask.sum())
        accepted_count = int((keep & mask).sum())
        report["rejection_by_class"][name] = {
            "total": total, "accepted": accepted_count, "rejected": total - accepted_count,
            "quality_rejected": int((mask & ~quality).sum()),
            "confidence_rejected": int((mask & quality & ~keep).sum()),
            "rejection_rate": 1 - accepted_count / total if total else None,
        }
    report["calibrated_confidence_quantiles"] = [float(v) for v in np.quantile(p.max(1), [0, .25, .5, .75, 1])]
    return report


def evaluate_cpu(runtime, rows):
    logits = np.concatenate([
        runtime.logits(np.stack([normalize_pixels(face) for face, _, _ in rows[i:i + MAX_BATCH]]))
        for i in range(0, len(rows), MAX_BATCH)
    ])
    return logits, np.array([label for _, label, _ in rows]), np.array([quality_reason(face) is None for face, _, _ in rows])


def finite_probe(model):
    outcomes = {}
    for name, invalid in (("NaN", np.nan), ("Inf", np.inf)):
        batch = np.full((1, 1, SIDE, SIDE), invalid, dtype=np.float32)
        try:
            model.logits(batch)
        except ValueError:
            outcomes[name] = "rejected"
        else:
            outcomes[name] = "accepted"
    return outcomes


def run_training(args, config, rows, baseline):
    variant = config["experiments"][args.experiment]
    started = perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    train = DataLoader(FaceDataset(rows["Training"], augment=True), batch_size=config["batch_size"], shuffle=True,
                       generator=torch.Generator().manual_seed(config["seed"]), num_workers=0)
    val = DataLoader(FaceDataset(rows["PublicTest"]), batch_size=config["batch_size"], num_workers=0)
    model = classifier(variant, config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=.0001)
    counts = np.bincount([label for _, label, _ in rows["Training"]], minlength=len(LABELS))
    weights = torch.tensor(counts, dtype=torch.float32).pow(-variant["imbalance_power"])
    loss_fn = nn.CrossEntropyLoss(weight=weights / weights.mean())
    best = -1.0
    best_epoch = 0
    checkpoint = args.out / "best.pt"
    write_report(args.out / "config.json", {"configuration": config, "experiment": args.experiment,
                 "prepared_manifest_sha256":sha_file(args.prepared/"dataset-manifest.json"),
                 "training_counts":counts.tolist(), "torch":torch.__version__, "device":"cpu",
                 "labels":"FER2013 original", "schedule":"cosine to 0.0001", "preprocessing":"lm-expression-gray48-v1"})
    for epoch in range(1, config["epochs"] + 1):
        loss, accuracy = train_epoch(model, train, optimizer, "cpu", loss_fn)
        logits, labels = collect(model, val, "cpu")
        val_scores = scores(probabilities(logits, 1), labels)
        score = val_scores["macro_f1"]
        print(json.dumps({"epoch": epoch, "loss":loss,"training_accuracy":accuracy,"macro_f1":score,
                          "balanced_accuracy":val_scores["balanced_accuracy"],"lr":scheduler.get_last_lr()[0]}), flush=True)
        scheduler.step()
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), checkpoint)
        elif epoch - best_epoch >= config["patience"]:
            break
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    logits, labels = collect(model, val, "cpu")
    quality = np.array([quality_reason(face) is None for face, _, _ in rows["PublicTest"]])
    policy, curve = calibration(logits, labels, quality, baseline["validation"], config["minimum_validation_accepted"])
    pending = args.out/"expression.pending.onnx"
    runtime, parity = export_and_check(model, rows["PublicTest"], config, checkpoint, policy, pending)
    finite = finite_probe(runtime)
    if finite != {"NaN":"rejected","Inf":"rejected"}:
        raise ValueError(f"Invalid ONNX input probe: {finite}")
    path = args.out/"expression.onnx"
    pending.rename(path)
    write_report(args.out/"validation.json", {
        "experiment":args.experiment,"best_epoch":best_epoch,"epochs_run":epoch,
        "checkpoint_sha256":sha_file(checkpoint),"checkpoint_bytes":checkpoint.stat().st_size,
        "onnx_sha256":sha_file(path),"onnx_bytes":path.stat().st_size,
        "parameters":sum(p.numel() for p in model.parameters()),"policy":asdict(policy),
        "validation":detailed(logits,labels,quality,policy),"accuracy_coverage_curve":curve,
        "parity":parity,"nonfinite_input":finite,"cpu_validation_latency":benchmark(runtime,rows["PublicTest"]),
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "seconds":perf_counter()-started,
    })


def validation_pass(candidate, baseline):
    raw, control = candidate["without_gating"], baseline["without_gating"]
    return (candidate["with_gating"] is not None
            and raw["macro_f1"] > control["macro_f1"]
            and raw["balanced_accuracy"] > control["balanced_accuracy"]
            and candidate["coverage"] > baseline["coverage"]
            and candidate["with_gating"]["accuracy"] >= baseline["with_gating"]["accuracy"]
            and raw["per_class"]["Disgust"]["recall"] >= control["per_class"]["Disgust"]["recall"]
            and all(raw["per_class"][label]["recall"] >= control["per_class"][label]["recall"] - .10
                    for label in LABELS))


def run_evaluation(args, config, rows, baseline):
    folders = {key:args.runs/f"expression-r9-{key}" for key in config["experiments"]}
    infos = {key:json.loads((folder/"validation.json").read_text()) for key,folder in folders.items()}
    for key, folder in folders.items():
        assert sha_file(folder/"expression.onnx") == infos[key]["onnx_sha256"]
        assert sha_file(folder/"best.pt") == infos[key]["checkpoint_sha256"]
        assert json.loads((folder/"config.json").read_text())["prepared_manifest_sha256"] == sha_file(args.prepared/"dataset-manifest.json")
    eligible = [key for key,value in infos.items() if validation_pass(value["validation"], baseline["validation"])]
    choice = max(eligible, key=lambda key:infos[key]["validation"]["without_gating"]["macro_f1"], default="E0")
    assert sha_file(args.baseline/"yolo26-expression.onnx") == baseline["onnx"]["sha256"]
    assert sha_file(args.baseline/"best.pt") == baseline["checkpoint_sha256"]
    args.out.mkdir(parents=True, exist_ok=False)
    frozen = {"selected_from_validation":choice,"eligible_from_validation": eligible,
              "condition":"Improve macro F1, balanced accuracy and coverage; no lower selective accuracy or Disgust recall; no class recall loss >10 pp"}
    write_report(args.out/"selection-before-test.json",frozen)
    output = {}
    for key, path in {"E0":args.baseline/"yolo26-expression.onnx",
                      **{k:f/"expression.onnx" for k,f in folders.items()}}.items():
        model = ExpressionModel.from_file(path)
        logits, labels, quality = evaluate_cpu(model, rows["PrivateTest"])
        output[key] = {"test":detailed(logits,labels,quality,model.policy),
                       "onnx_sha256":sha_file(path),"checkpoint_sha256":model.version,
                       "onnx_bytes":path.stat().st_size,"cpu_latency":benchmark(model,rows["PrivateTest"]),
                       "peak_rss_kib_process":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    promoted = choice if choice != "E0" and validation_pass(output[choice]["test"], output["E0"]["test"]) else "E0"
    write_report(args.out/"evaluation.json",{"validation_selection":frozen,"active_after_holdout_guardrail":promoted,
                 "heldout_guardrail":"Reject the already selected candidate on a regression; no reranking on PrivateTest",
                 "results":output})
    print("VALIDATION_SELECTION",choice,"ACTIVE",promoted,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E1","E2","E3"))
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", type=Path)
    args = parser.parse_args()
    if (args.experiment is None) == (args.runs is None):
        parser.error("Choose --experiment for training or --runs for frozen evaluation")
    config = json.loads(CONFIG.read_text())
    setup(config["seed"])
    rows, _ = load_prepared(args.prepared)
    baseline = json.loads((args.baseline/"evaluation.json").read_text())
    if args.experiment:
        run_training(args,config,rows,baseline)
    else:
        run_evaluation(args,config,rows,baseline)


if __name__ == "__main__":
    main()
