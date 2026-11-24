# SPDX-License-Identifier: AGPL-3.0-only
# Extracted/adapted for standalone Core CV, 2026-11-24. No product service code.
"""Export a user-supplied, face-fine-tuned YOLO26 detect checkpoint to ONNX.

Run only in a separate training/export environment with Ultralytics installed.
This script deliberately refuses stock COCO weights (classes != {0: 'face'}).
Does not train, download or change the inference server.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fine-tuned YOLO26 face model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    destination = args.destination.expanduser().resolve()
    if checkpoint.suffix.lower() != ".pt":
        parser.error("--checkpoint must be a .pt file")
    if destination.suffix.lower() != ".onnx":
        parser.error("--destination must end with .onnx")
    if destination.exists():
        parser.error("Refusing to overwrite an existing ONNX artifact")
    if not destination.parent.is_dir():
        parser.error("Destination directory does not exist")

    from ultralytics import YOLO

    from onnx import TensorProto, checker, load

    # Ultralytics writes the ONNX beside the input. Keep that write off the source tree.
    with TemporaryDirectory() as folder:
        staged = Path(folder) / checkpoint.name
        shutil.copyfile(checkpoint, staged)
        model = YOLO(str(staged))
        if model.task != "detect" or model.names != {0: "face"}:
            parser.error("Checkpoint must be a detect model with exactly {0: 'face'}")
        exported = Path(model.export(
            format="onnx", imgsz=640, batch=1, dynamic=False,
            nms=False, device="cpu", simplify=False,
        )).resolve()
        graph = load(exported, load_external_data=False)
        checker.check_model(graph)
        inputs, outputs = graph.graph.input, graph.graph.output
        expected = ([1, 3, 640, 640], [1, 300, 6])
        if len(inputs) != 1 or len(outputs) != 1 or any(
            node.type.tensor_type.elem_type != TensorProto.FLOAT
            or [dim.dim_value for dim in node.type.tensor_type.shape.dim] != shape
            for node, shape in zip((inputs[0], outputs[0]), expected)
        ):
            raise RuntimeError("Export must produce static float32 [1,3,640,640] to [1,300,6]")
        if any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
            raise RuntimeError("Export unexpectedly contains NMS")
        with exported.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    print(f"EXPORTED: {destination}")
    print("Next: validate tensor contracts, parity and release rights before activation")


if __name__ == "__main__":
    main()
