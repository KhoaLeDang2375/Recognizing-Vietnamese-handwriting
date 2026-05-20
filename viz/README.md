# viz/ — Vietnamese Handwriting OCR (Gradio + React)

A self-contained, modern frontend for the same SVTR / CRNN inference pipeline
that powers `../app.py`. The Streamlit demo is left untouched as the
reference implementation; this directory replaces the UI and the ngrok-based
deployment story.

```
viz/
├── backend/
│   ├── inference.py      # 5 core functions copied verbatim from app.py
│   ├── server.py         # FastAPI + Gradio (api_name="infer"); share tunnel
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/              # React + TS strict + Tailwind + framer-motion
    └── package.json
```

The five inference primitives (`MODEL_MAP`, `adaptive_crop_text_region`,
`adaptive_preprocess_for_ocr`, `parse_infer_output`, `run_inference`) are
**copied byte-for-byte** from `app.py` to guarantee reproducibility. The same
env-var contract is respected: `PADDLEOCR_DIR`, `WORK_DIR`, `DICT_PATH`,
`SVTR_CKPT`, `CRNN_CKPT`, `SVTR_CFG`, `CRNN_CFG`, `TEMP_DIR`.

---

## Mode A — Local dev (two processes)

Use this when iterating on the UI. Vite gives you HMR; the backend serves the
Gradio API on a separate port.

```bash
# 1. Install backend deps (assumes PaddlePaddle + PaddleOCR are already set up,
#    same as for app.py).
pip install -r viz/backend/requirements.txt

# 2. Point at your checkpoints (or `cp viz/backend/.env.example .env` and edit).
export PADDLEOCR_DIR=./PaddleOCR
export DICT_PATH=./vietnamses_dict.txt
export SVTR_CKPT=/path/to/svtr/best_accuracy
export CRNN_CKPT=/path/to/crnn/best_accuracy
export SVTR_CFG=/path/to/rec_svtr_stage2.yml
export CRNN_CFG=/path/to/rec_crnn_stage2.yml

# 3. Start Gradio backend (terminal 1)
python viz/backend/server.py
#   → http://localhost:7860/gradio
#   → http://localhost:7860/healthz

# 4. Start Vite dev server (terminal 2)
cd viz/frontend
pnpm install         # or: npm install
pnpm dev             # http://localhost:5173
```

`viz/frontend/.env.development` sets `VITE_GRADIO_URL=http://localhost:7860/gradio`
so the SPA talks to the local Gradio mount. CORS for `localhost:5173` is
pre-configured in `server.py` — override with `VIZ_CORS_ORIGINS=...` if needed.

---

## Mode B — Notebook (Kaggle / Colab, single process, single URL)

Use this in `demo-streamlit-ui.ipynb`-style workflows. One Python process
serves both the React SPA and the Gradio API, and a single `.gradio.live`
tunnel exposes everything.

```bash
# 1. Build the SPA once.
cd viz/frontend
pnpm install
pnpm build           # → viz/frontend/dist/

# 2. Install backend deps (same as Mode A).
cd ..
pip install -r backend/requirements.txt

# 3. Launch with share=True.
GRADIO_SHARE=1 python backend/server.py
```

Output:

```
Public tunnel : https://<random>.gradio.live
SPA (public)  : https://<random>.gradio.live/
Gradio (pub.) : https://<random>.gradio.live/gradio
```

In production the frontend leaves `VITE_GRADIO_URL` unset (see
`.env.production`); `resolveGradioUrl()` falls back to
`window.location.origin + "/gradio"`, so the SPA correctly targets whichever
`.gradio.live` subdomain Gradio happens to grab.

### Notebook snippet

Replace the ngrok cell in `demo-streamlit-ui.ipynb` with:

```python
import os, sys, subprocess, time
from pathlib import Path

VIZ = Path('/kaggle/working/Recognizing-Vietnamese-handwriting/viz')

# Build SPA once per session
subprocess.check_call(['pnpm', 'install'], cwd=str(VIZ / 'frontend'))
subprocess.check_call(['pnpm', 'build'],   cwd=str(VIZ / 'frontend'))

env = os.environ.copy()
env.update({
    'PADDLEOCR_DIR': str(PADDLEOCR_DIR),
    'WORK_DIR':      str(WORK_DIR),
    'DICT_PATH':     str(DICT_PATH),
    'SVTR_CKPT':     str(SVTR_CKPT),
    'CRNN_CKPT':     str(CRNN_CKPT),
    'SVTR_CFG':      str(SVTR_CFG),
    'CRNN_CFG':      str(CRNN_CFG),
    'TEMP_DIR':      str(TEMP_DIR),
    'GRADIO_SHARE':  '1',
})

proc = subprocess.Popen(
    [sys.executable, str(VIZ / 'backend' / 'server.py')],
    env=env,
)
time.sleep(8)   # give it a moment to log the public URL
```

No `pyngrok`, no `NGROK_TOKEN`. The Gradio share tunnel handles it.

---

## API contract

The frontend talks to one Gradio endpoint:

```ts
// POST {GRADIO}/run/predict   (handled by @gradio/client)
client.predict("/infer", [imageBlob, modelChoice, usePreprocess])
```

Returns:

```jsonc
{
  "ok": true,
  "results": [{ "text": "Tôi yêu Việt Nam", "conf": 0.9821 }],
  "elapsed": 1.42,
  "raw": "...full stdout+stderr of infer_rec.py...",
  "model": "SVTR",
  "preprocessed": false,
  "error": null
}
```

A FastAPI `/healthz` is also exposed (handy for liveness probes / smoke tests).

---

## What's intentionally absent

To stay true to the existing demo, this UI does **not** add: model
comparison, history, batch upload, before/after slider, auth, dark mode,
persistent storage, queues, or databases.
