// SPDX-License-Identifier: AGPL-3.0-only
// Integrity pins identify the local candidates; they do not grant publication rights.
export const MODEL_PINS = Object.freeze({
  faceDetector: { source: 'yolo26n-face.onnx', file: 'yolo26-face-6f71d6097fa024ce.onnx', contract: 'yolo26-face-v1',
    bytes: 9757238, sha256: '6f71d6097fa024cec3bdc60ccb0e9c5dae79e422a6af8380a57ff5e974bbbc5b' },
  expression: { source: 'yolo26-expression-review9-E1.onnx', file: 'cnn-e1-1e2e48b8a08e551b.onnx', contract: 'e1-gray48-v1',
    bytes: 855336, sha256: '1e2e48b8a08e551b87e2727ae1d6311224e940ffa6ef6858842804c1be324b1d' },
  landmarks: { source: 'landmarks-qualcomm-v0.64.0.onnx', file: 'mesh468-ac4525a414998f3c.onnx', contract: 'facemesh468-v1',
    bytes: 2706699, sha256: 'ac4525a414998f3cf35e3eeed6b76c5f842b7ff4e308f8ece4a4802ac5bac30a' },
});
