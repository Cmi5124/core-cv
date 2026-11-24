# SPDX-License-Identifier: AGPL-3.0-only
# Extracted/adapted for standalone Core CV, 2026-11-24. No product service code.
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Timer
import numpy as np
from PIL import Image
LABELS = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
SIDE = 48
MAX_BATCH = 4
CONTRACT = "lm-expression-gray48-v1"

def expression_pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L").resize((SIDE, SIDE), Image.Resampling.BILINEAR))

def normalize_pixels(pixels: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(pixels.astype(np.float32)[None] / np.float32(255))

def quality_reason(pixels: np.ndarray) -> str | None:
    values = pixels.astype(np.float32)
    if values.std() < 9:
        return "low_contrast"
    center = values[1:-1, 1:-1]
    laplacian = values[:-2, 1:-1] + values[2:, 1:-1] + values[1:-1, :-2] + values[1:-1, 2:] - 4 * center
    return "blurred" if laplacian.var() < 18 else None

def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    weights = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)

@dataclass(frozen=True)
class ExpressionPolicy:
    temperature: float
    min_probability: float | None
    min_margin: float

    @classmethod
    def from_json(cls, value: str) -> ExpressionPolicy:
        data = json.loads(value)
        if not isinstance(data, dict) or set(data) != {"temperature", "min_probability", "min_margin"}:
            raise ValueError("Invalid expression policy")

        def number(key: str, minimum: float, maximum: float) -> float:
            raw = data[key]
            if type(raw) not in (float, int) or not minimum <= raw <= maximum:
                raise ValueError("Invalid expression policy value")
            return float(raw)

        threshold = None if data["min_probability"] is None else number("min_probability", 1 / len(LABELS), 1)
        return cls(number("temperature", 0.1, 10), threshold, number("min_margin", 0, 1))

def accepted(probability: np.ndarray, policy: ExpressionPolicy) -> np.ndarray:
    top = np.sort(probability, axis=-1)[:, -2:]
    if policy.min_probability is None:
        return np.zeros(len(top), dtype=bool)
    return (top[:, 1] >= policy.min_probability) & (top[:, 1] - top[:, 0] >= policy.min_margin)

def read_model(path: Path, limit: int) -> bytes:
    # Read the self-contained model once so the checked bytes are the bytes ORT loads.
    if not path.is_file():
        raise FileNotFoundError("Expression model file is unavailable")
    with path.open("rb") as stream:
        model = stream.read(limit + 1)
    if not model or len(model) > limit:
        raise ValueError("Invalid expression model size")
    return model

def cpu_session(model: bytes):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    return session, ort.RunOptions

def checkpoint_version(meta: dict[str, str], contract: str) -> str:
    version = meta.get("checkpoint_sha256", "")
    if (meta.get("contract") != contract or json.loads(meta.get("labels", "null")) != list(LABELS)
            or not re.fullmatch(r"[0-9a-f]{64}", version)):
        raise ValueError("Invalid expression model metadata")
    return version

class ExpressionModel:
    input_name = "faces"
    output_name = "logits"

    def __init__(self, session, run_options, policy, version):
        self.session = session
        self.run_options = run_options
        self.policy = policy
        self.version = version

    @classmethod
    def from_file(cls, path: Path) -> ExpressionModel:
        session, run_options = cpu_session(read_model(path, 8 * 1024 * 1024))
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "faces" or inputs[0].type != "tensor(float)"
                or inputs[0].shape != ["batch", 1, SIDE, SIDE] or len(outputs) != 1
                or outputs[0].name != "logits" or outputs[0].type != "tensor(float)"
                or outputs[0].shape != ["batch", len(LABELS)]):
            raise ValueError("Invalid expression ONNX tensor contract")
        meta = session.get_modelmeta().custom_metadata_map
        version = checkpoint_version(meta, CONTRACT)
        result = cls(session, run_options, ExpressionPolicy.from_json(meta.get("policy", "{}")), version)
        for batch in (1, MAX_BATCH):
            result.logits(np.zeros((batch, 1, SIDE, SIDE), dtype=np.float32), timeout=1)
        return result

    def logits(self, batch: np.ndarray, timeout: float = 0.2) -> np.ndarray:
        options = self.run_options()
        options.log_severity_level = 4
        timer = Timer(timeout, lambda: setattr(options, "terminate", True))
        timer.daemon = True
        timer.start()
        try:
            values, = self.session.run([self.output_name], {self.input_name: batch}, options)
        finally:
            timer.cancel()
        if options.terminate:
            raise TimeoutError("Expression inference timed out")
        if values.dtype != np.float32 or values.shape != (len(batch), len(LABELS)) or not np.isfinite(values).all():
            raise ValueError("Invalid expression logits")
        return values
