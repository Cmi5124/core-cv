# SPDX-License-Identifier: AGPL-3.0-only
# Extracted/adapted for standalone Core CV, 2026-11-24. No product service code.
from pathlib import Path
from dataclasses import asdict
from time import perf_counter
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from runtime import CONTRACT, LABELS, MAX_BATCH, SIDE, ExpressionModel, expression_pixels, normalize_pixels
from dataset import sha_file
from prepare import load_prepared

class FaceDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        pixels, label, _ = self.rows[index]
        image = Image.fromarray(pixels)
        if self.augment:
            if torch.rand(()).item() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            x, y = torch.randint(-2, 3, (2,)).tolist()
            image = image.transform(image.size, Image.Transform.AFFINE, (1, 0, x, 0, 1, y),
                                    resample=Image.Resampling.BILINEAR, fillcolor=114)
        return torch.from_numpy(normalize_pixels(expression_pixels(image))), label

def collect(model, loader, device):
    logits, targets = [], []
    model.eval()
    with torch.inference_mode():
        for example, label in loader:
            logits.append(model(example.to(device)).cpu().numpy())
            targets.append(label.numpy())
    return np.concatenate(logits), np.concatenate(targets)

def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    loss_sum, correct = 0.0, 0
    for example, label in loader:
        example, label = example.to(device), label.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(example)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(label)
        correct += int((output.argmax(1) == label).sum().item())
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)

def export_and_check(model, rows, config, checkpoint, policy, path):
    import onnx
    import onnxruntime as ort

    model.cpu().eval()
    sample = torch.stack([FaceDataset(rows)[i][0] for i in range(MAX_BATCH)])
    torch.onnx.export(model, sample, str(path), dynamo=False, opset_version=config["onnx_opset"],
                      input_names=["faces"], output_names=["logits"],
                      dynamic_axes={"faces": {0: "batch"}, "logits": {0: "batch"}})
    graph = onnx.load(path)
    onnx.helper.set_model_props(graph, {"contract": CONTRACT, "labels": json.dumps(LABELS),
                                       "checkpoint_sha256": sha_file(checkpoint), "policy": json.dumps(asdict(policy))})
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, path)
    runtime = ExpressionModel.from_file(path)
    tested = []
    with torch.inference_mode():
        for index in range(0, min(len(rows), 32), MAX_BATCH):
            examples = torch.stack([FaceDataset(rows)[i][0] for i in range(index, min(index + MAX_BATCH, len(rows)))])
            for batch in (examples[:1], examples):
                expected = model(batch).numpy()
                actual = runtime.logits(batch.numpy())
                repeated = runtime.logits(batch.numpy())
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                np.testing.assert_array_equal(actual, repeated)
                tested.append(float(np.abs(actual - expected).max()))
    try:
        runtime.session.run(["logits"], {"faces": np.zeros((1, 3, SIDE, SIDE), dtype=np.float32)})
    except ort.capi.onnxruntime_pybind11_state.InvalidArgument:
        pass
    else:
        raise ValueError("ONNX accepted an invalid input shape")
    return runtime, {"rtol": 1e-4, "atol": 1e-5, "comparisons": len(tested),
                     "max_abs_diff": max(tested), "invalid_input_rejected": True, "repeat_exact": True}

def benchmark(runtime, rows):
    samples = [FaceDataset(rows)[i][0].numpy()[None] for i in range(min(100, len(rows)))]
    for row in samples[:10]:
        runtime.logits(row)
    measures = []
    for row in samples:
        start = perf_counter()
        runtime.logits(row)
        measures.append((perf_counter() - start) * 1000)
    return {"batch": 1, "n": len(measures),
            "p50_ms": float(np.percentile(measures, 50)), "p95_ms": float(np.percentile(measures, 95))}

def write_report(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
