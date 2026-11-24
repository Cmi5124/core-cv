# Core CV notices

Copyright 2026 CmiTeLab. Authored Core CV source is licensed under AGPL-3.0-only; see LICENSE. The browser implementation and standalone extraction were created/modified on 2026-11-24. No warranty. The repository contains no product backend, account integration, user data or model weights.

- ONNX Runtime Web 1.31.1, Microsoft Corporation, MIT. See `licenses/ONNX-Runtime-LICENSE.txt` and https://github.com/microsoft/onnxruntime. npm's package declares MIT but omits the full native third-party notices. The official LICENSE/ThirdPartyNotices endpoints returned 404 during this implementation; the official indexed MIT license was available. Reconcile and ship the notices matching the actual WASM binary before any binary website release. The source candidate does not include the generated runtime binaries.
- Vite 8.3.6 is an unmodified MIT build tool, resolved by package-lock.json; retain its package notices if redistributing the tool.
- `src/pixels.js` implements resampling compatible with Pillow's separable bilinear method and 22-bit coefficient rounding. Keep the Pillow notice in `licenses/Pillow-LICENSE.txt`. Reference: https://github.com/python-pillow/Pillow/blob/12.2.0/src/libImaging/Resample.c. The Python oracle used Pillow 12.2.0.
- The float32 quality reductions follow NumPy 2.4.6's pairwise summation order, adapted to JavaScript on 2026-11-25; see `licenses/NumPy-LICENSE.txt` (BSD-3-Clause) and https://github.com/numpy/numpy/blob/v2.4.6/numpy/_core/src/umath/loops_utils.h.src. NumPy is not a browser dependency.
- Model licensing is separate from this source license. No third-party weights, dataset, external tensors or model code copied from Qualcomm/Ultralytics is included. Model-specific attribution and unresolved release evidence are in MODEL_CARD.md. The YOLO export wrapper calls an existing Ultralytics installation; it does not vendor that library.

Dependency licenses remain their respective upstream licenses. The AGPL source license does not grant third-party dataset or checkpoint rights.
