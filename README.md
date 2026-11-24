# Core CV — standalone browser lab

An independent camera application with YOLO26 face detection, CNN E1 expression classification and Qualcomm 468-point mesh/geometry. Authored source: **AGPL-3.0-only**. Only Core CV belongs in its separate public repository. Other CmiTeLab applications, accounts, APIs, deployment credentials, data and monorepo history are outside this source package.

The full three-model pipeline is implemented and verified with local candidates. **Weights are not published or included in a build.** The public manifest is deliberately empty until artifact rights/source/notices are resolved. See [MODEL_CARD.md](MODEL_CARD.md), [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).

## Local development

Node 24 was used. The app reuses onnxruntime-web 1.31.1 and Vite 8.3.6. No Python dependency is needed to run the browser app.

```powershell
npm ci --ignore-scripts
# Optional: an existing directory containing the three authorized local candidates.
$env:CORE_CV_LOCAL_MODEL_DIR = 'D:\authorized-models'
npm run dev
```

Open `http://127.0.0.1:5186/`. Camera selection and all three stages are automatic after **Đồng ý & bắt đầu**. Stop, revoke, hide the tab or close the page to release capture and inference. No frames/results leave the device, and no model is silently replaced with a remote service or simulator.

The optional dev directory must contain the exact filenames/bytes/SHA values in `tools/model-specs.mjs`. Without this variable, the dev route checks `local-models/` beside package.json. This loopback-only route rejects foreign Origin/Host/cross-site requests and never copies models into `public/` or `dist/`. Restart the dev server after replacing a candidate; successfully verified bytes are retained for that dev process.

```powershell
npm test
npm run build
npm run preview
```

The production preview at `http://127.0.0.1:5187/` intentionally reports unpublished models and keeps the camera off. `public/_headers` supplies CSP, camera-only permissions and immutable caching for versioned assets; the manifest is no-store. Use an independent origin and deploy only `dist/` after the release gates below.

## Reproduction and verification

`tests/fixtures/oracle.json` contains deterministic synthetic inputs/results; it contains no real face images or private records. The Node suite covers pixel parity, calibrated probabilities, geometry, rejected tensors, face budgets, loader integrity/size/origin/abort, Worker/session lifecycle, camera denial/revoke and source export boundaries.

With the three existing local models, open `/tests/browser-harness.html` on the dev server and click **Run WASM parity**. This actually executes the three ONNX models and a real Worker against the recorded Python CPU reference. It is a developer test page, excluded from the production build. Synthetic parity is not an accuracy benchmark.

`reproduce/expression/` contains the actual residual E1 architecture and isolated preparation/training/export/calibration source adapted from the original experiment on 2026-11-24. It excludes the private product runtime and hybrid optimizations. The training path uses original FER2013 labels. In an **existing Linux training environment** with authorized local data, NumPy/Pillow/PyTorch/ONNX/ONNX Runtime installed:

```sh
cd reproduce/expression
python prepare.py --csv /authorized/fer2013.csv --provenance /authorized/provenance.json --out /local/prepared
python review9.py --experiment E1 --prepared /local/prepared --baseline /local/baseline --out /local/expression-r9-E1
```

The provenance JSON must record the official `source_url`, `terms_url`, `authorized_use`, timezone-aware `downloaded_at` and exact raw-file `sha256`; see `prepare.py` for validation. Baseline directory supplies `evaluation.json` from the authorized baseline experiment; calibration selects against its validation accuracy. Retraining can change weights and the calibrated policy; do not silently substitute it for the frozen browser profile. Training uses Linux `resource`, was not rerun here, and is not claimed byte-reproducible. No data/checkpoints are provided or downloaded.

The historical baseline evaluation report is not included. It is an external prerequisite for this training command, so this recipe is not a self-contained reconstruction of the historical checkpoint. The standalone browser source/build does not require it.

`reproduce/export_yolo26_face.py` exports an existing authorized face-only Ultralytics checkpoint; it does not recover missing upstream training history. `reproduce/pack_mesh.py` embeds an existing Qualcomm graph's external tensors and reports whether byte identity matches. Use the pinned source ZIP/card/attribution in MODEL_CARD.md and an existing ONNX environment. Neither script publishes artifacts. Training/export dependencies are optional existing tools; this project has not installed them.

With an existing Python CPU inference environment, run `python -B -m unittest discover -s reproduce -p test_runtime.py -v`; set `CORE_CV_LOCAL_MODEL_DIR` to enable the exact E1 checkpoint comparison. Without it, that inference case is explicitly skipped.

## Public source export

```sh
node tools/export-source.mjs /absolute/new/core-cv-source-candidate
```

This creates a new directory from an explicit path allowlist and writes SOURCE_MANIFEST.json with file SHA-256 values. It refuses existing destinations, traversal, symlinks, environment files, weights and detected secrets/private imports. It does not copy Git history, other applications, node_modules, generated runtime binaries, local data or dist. Inspect the candidate, run a clean install/test/build there, record its digest and use the source publishing gate before creating the separate GitHub repo. Never make the private monorepo public.

Only an ordinary outbound link may open this standalone app from other private applications. No iframe, imported Worker, SDK bridge, account/session or result exchange is implemented. AGPL scope depends on the actual combined work; a separate repo alone does not establish that boundary.

## Release gates and rollback

1. Resolve the exact YOLO26-Face and E1 artifact rights and missing upstream corresponding source; retain Qualcomm and runtime notices matching the distributed binaries. `MODEL_CARD.md` records the current evidence gaps.
2. Review the allowlisted source, record the publishing gate result, and publish it to a new dedicated repository. Verify the source URL works before distributing a website binary. The visible source page must identify the exact deployed source revision.
3. Validate a manifest of all three approved files, their contracts/byte counts/hashes, real-camera behavior and target browsers in an independent-origin preview. Do not activate a null/incomplete manifest in private product links.
4. Deploy only the reviewed static directory, verify cache/CSP/permissions/source offer and actual inference; retain the previous Pages deployment ID for rollback. Stop deployment if any source/model gate fails.

Current verification target is desktop Chromium. Physical mobile devices, positive face/mesh accuracy, sustained throughput and full retraining/repacking remain unverified. The existing private app is preserved while public model distribution is unresolved.
