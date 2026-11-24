# Core CV 0.1 model profile

Status: the three pinned candidates execute locally in ONNX Runtime Web 1.31.1 WASM. Public weights are **not released**. `public/models/core-cv.json` intentionally contains three null entries. The loopback dev server checks existing local files; the production build never copies them.

| Role | Local candidate | SHA-256 | Public artifact status |
| --- | --- | --- | --- |
| YOLO26 nano face detector | yolo26n-face.onnx, 9,757,238 bytes | 6f71d6097fa024cec3bdc60ccb0e9c5dae79e422a6af8380a57ff5e974bbbc5b | Not cleared |
| CNN residual E1 | yolo26-expression-review9-E1.onnx, 855,336 bytes | 1e2e48b8a08e551b87e2727ae1d6311224e940ffa6ef6858842804c1be324b1d | Not cleared |
| Qualcomm mesh 468 | landmarks-qualcomm-v0.64.0.onnx, 2,706,699 bytes | ac4525a414998f3cf35e3eeed6b76c5f842b7ff4e308f8ece4a4802ac5bac30a | Positive Apache-2.0 evidence; release notices/packaging still required |

YOLO source: [theminji/yolo26n-face-detection, pinned revision](https://huggingface.co/theminji/yolo26n-face-detection/tree/19ea41600eb776e223171f24ebfcb9fc9854d8bf). The card declares AGPL-3.0. Original PT hash: `8e8d80ad2f17ec8e2505ee5cdf735e06908ade959ab68a95d3d11d27b98d17ca`. The linked Roboflow WIDER mirror declares CC BY 4.0, whereas the [CUHK dataset card](https://huggingface.co/datasets/CUHK-CSE/wider_face) declares CC BY-NC-ND 4.0. No evidence reviewed establishes the mirror's authority to broaden these rights or the exact fine-tune's full corresponding training source. Public checkpoint distribution remains unresolved; this is not a determination that weights necessarily derive copyright from images. Export code is supplied, but it cannot reconstruct missing upstream training history.

E1 is a custom residual CNN, **not a YOLO expression head**, despite its historical filename. The exact architecture, preparation, training, calibration and ONNX export source are under `reproduce/expression/`. Training used **original FER2013 labels**, not FER+ annotations. Checkpoint hash: `c546b469b4ab815fc83b28a2c7b695cb8b9bae20deb846ebf2aa16ea914752cc`. [FER2013 competition terms](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/rules) have not established public distribution rights for this particular E1 checkpoint. No separate grant was supplied. FER+ MIT annotations would not by themselves clear the underlying FER2013 images/checkpoint.

Mesh source: [Qualcomm MediaPipe Face model card](https://huggingface.co/qualcomm/MediaPipe-Face-Detection), [source v0.64.0 commit](https://github.com/qualcomm/ai-hub-models/tree/53af617346c1a9c51c0cdf9d05540946d02741e5/src/qai_hub_models/models/mediapipe_face). The model card declares Apache-2.0; Qualcomm source separately uses BSD-3-Clause. [Original ZIP](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mediapipe_face/releases/v0.64.0/mediapipe_face-onnx-float.zip) SHA `f6d60fdf53ccf6fa2ce1f3b103fcc468b6860b4e707223a0a6d6733b35e0c630`. Graph SHA `ebd4d22b162cfb3f8113fa48908859a4413c45a82a9ae0bd8a1e5a56f1fb9c5f`; external tensor SHA `7b4be428f66b16768aecb31954848ec9512eb444f13eaf6fc4690acc981d9732`. Local packaging embeds external tensors; retain upstream attribution, license and any NOTICE when distributing the artifact. No Qualcomm source or weights is vendored here.

## Runtime behavior

- YOLO: float32 RGB CHW /255, centered letterbox 640, output `[1,300,6]`, class 0, confidence ≥0.4, maximum 20 boxes.
- E1: direct face crop, grayscale then Pillow-compatible bilinear 48×48, /255. Reject clipped/small/flat/blurred crops. At most four eligible faces per pass and 5Hz ceiling. Labels: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. Frozen temperature **1.231144413344916**, probability **0.63**, margin **0.05**. Otherwise Unknown. No hybrid/C2/autocontrast/geometry fusion.
- Mesh: 1.12 square ROI, minimum side 52, RGB192 /255, presence ≥0.5; at most two faces and 2Hz. Validate geometry before returning 12 measurements. No iris, blendshapes, FACS, attention or learning-state predictions.
- One WASM thread in one Worker, no queued frames, 60s load deadline and 3s frame deadline. Actual frame cadence depends on the device. No server fallback, frame upload or saved capture history.

## Evidence and limits

Synthetic Pillow preprocessing fixtures match byte-for-byte. Four E1 synthetic inputs, 1,404 mesh values and 72 YOLO output values were compared between Python CPU ORT 1.26.0 and browser WASM ORT 1.31.1; categorical outcomes agree. The real Worker loaded all three sessions, inferred, stopped and restarted. Synthetic blank/pattern inputs correctly produced no faces/low mesh presence. These checks establish execution/parity, **not face accuracy, emotion validity, demographic fairness or clinical/educational suitability**. No physical mobile-device, positive face/mesh or sustained FPS benchmark has been completed. Full reproduction training and mesh repacking were not rerun in this environment.

Public activation requires resolved artifact rights and corresponding source, matching dependency/model notices, target-device camera tests, reviewed hashes, the source publishing gate and a working independent-origin preview. A null manifest must keep the camera off and explain the missing release.
