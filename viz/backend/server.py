"""Gradio + FastAPI server for the Vietnamese Handwriting OCR pipeline.

One file, two modes:

  Mode A (local dev, 2 processes):
      python viz/backend/server.py
      → Gradio API at  http://localhost:7860/gradio
      → Vite SPA at    http://localhost:5173  (started separately)

  Mode B (notebook, 1 process, 1 URL):
      GRADIO_SHARE=1 python viz/backend/server.py
      → SPA + Gradio API behind a single  *.gradio.live  tunnel
      → SPA mounted at /   ; Gradio API at /gradio
"""

from __future__ import annotations

import base64
import io
import os
import secrets
import sys
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference import (
    MODEL_MAP,
    TEMP_DIR,
    adaptive_preprocess_for_ocr,
    run_inference,
)

# ── Spell Correction Model ──────────────────────────────────────────────────
print("[viz] Loading spell correction model 'bmd1905/vietnamese-correction-v2'...")
try:
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    corrector = pipeline(
        "text2text-generation",
        model="bmd1905/vietnamese-correction-v2",
        device=device
    )
    print(f"[viz] Spell correction model loaded successfully on device: {device}")
except Exception as e:
    print(f"[viz] Could not load spell correction model: {e}", file=sys.stderr)
    corrector = None


def spell_correct(raw_text: str) -> str:
    """Correct Vietnamese spelling errors in batch using BARTpho, with detailed logging."""
    import time
    
    print("-" * 40, flush=True)
    print(f"[viz][spell_correct] Received request.", flush=True)
    print(f"[viz][spell_correct] Input text:\n{raw_text!r}", flush=True)
    
    if not raw_text or not raw_text.strip():
        print("[viz][spell_correct] Input is empty or whitespace.", flush=True)
        print("-" * 40, flush=True)
        return raw_text
        
    if corrector is None:
        print("[viz][spell_correct] Corrector model is not loaded (corrector is None)!", flush=True)
        print("-" * 40, flush=True)
        return raw_text + " (Spell correction model not loaded)"

    t_start = time.time()
    lines = raw_text.splitlines()
    print(f"[viz][spell_correct] Total lines split: {len(lines)}", flush=True)
    
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    non_empty_lines = [lines[i] for i in non_empty_indices]
    
    if not non_empty_lines:
        print("[viz][spell_correct] No non-empty lines to process.", flush=True)
        print("-" * 40, flush=True)
        return raw_text

    print(f"[viz][spell_correct] Processing {len(non_empty_lines)} non-empty lines...", flush=True)
    for idx, line in enumerate(non_empty_lines):
        print(f"  Line {idx+1}/{len(non_empty_lines)}: {line!r}", flush=True)

    try:
        t0 = time.time()
        # Call the pipeline on the batch
        batch_results = corrector(non_empty_lines, max_length=256, batch_size=len(non_empty_lines))
        elapsed = time.time() - t0
        print(f"[viz][spell_correct] Model execution completed in {elapsed:.4f} seconds.", flush=True)
        
        for idx, res_idx in enumerate(non_empty_indices):
            original = lines[res_idx]
            corrected = batch_results[idx]["generated_text"]
            print(f"  Line {idx+1} result: {original!r} -> {corrected!r}", flush=True)
            lines[res_idx] = corrected
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"[viz][spell_correct] ERROR during batch prediction (elapsed {elapsed:.4f}s): {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        
    final_text = "\n".join(lines)
    print(f"[viz][spell_correct] Total time elapsed: {time.time() - t_start:.4f} seconds.", flush=True)
    print("-" * 40, flush=True)
    return final_text



# ── Paths & config ──────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
FRONTEND_DIST = (HERE.parent / "frontend" / "dist").resolve()

HOST = os.environ.get("VIZ_HOST", "0.0.0.0")
PORT = int(os.environ.get("VIZ_PORT", "7860"))
SHARE = os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes"}

# CORS origins for Mode A. Override with VIZ_CORS_ORIGINS="https://foo,https://bar".
_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("VIZ_CORS_ORIGINS", ",".join(_default_origins)).split(",")
    if o.strip()
]


# ── Inference adapter for Gradio ────────────────────────────────────────────
def _encode_preview(img: Image.Image, max_w: int = 1000) -> str:
    """Encode the image actually fed to the model as a compact JPEG data URL.

    Downscaled to `max_w` so the response stays small even for phone photos.
    """
    im = img.convert("RGB")
    if im.width > max_w:
        new_h = round(im.height * max_w / im.width)
        im = im.resize((max_w, new_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def infer(
    image: Image.Image | None,
    model_choice: str,
    use_preprocess: bool,
) -> dict[str, Any]:
    """Run OCR on one image. Returned dict is what the JS client consumes.

    Shape:
        {
          "ok": bool,
          "results": [{"text": str, "conf": float}, ...],
          "elapsed": float,           # seconds
          "raw": str,                 # raw stdout+stderr from infer_rec.py
          "model": str,               # algo name (SVTR / CRNN)
          "preprocessed": bool,
          "processed_image": str | None,  # data URL of the image fed to model
          "error": str | None,
        }
    """
    if image is None:
        return {
            "ok": False,
            "results": [],
            "elapsed": 0.0,
            "raw": "",
            "model": "",
            "preprocessed": False,
            "processed_image": None,
            "error": "No image provided.",
        }
    if model_choice not in MODEL_MAP:
        return {
            "ok": False,
            "results": [],
            "elapsed": 0.0,
            "raw": "",
            "model": "",
            "preprocessed": False,
            "processed_image": None,
            "error": f"Unknown model: {model_choice!r}",
        }

    img_rgb = image.convert("RGB")
    img_to_save = adaptive_preprocess_for_ocr(img_rgb) if use_preprocess else img_rgb
    processed_preview = _encode_preview(img_to_save)

    os.makedirs(TEMP_DIR, exist_ok=True)
    tmp_path = os.path.join(TEMP_DIR, "temp_infer.jpg")
    img_to_save.save(tmp_path, format="JPEG")

    try:
        results, elapsed, raw_out = run_inference(tmp_path, model_choice)
    except Exception as e:
        return {
            "ok": False,
            "results": [],
            "elapsed": 0.0,
            "raw": traceback.format_exc(),
            "model": MODEL_MAP[model_choice]["algo"],
            "preprocessed": bool(use_preprocess),
            "processed_image": processed_preview,
            "error": str(e),
        }

    return {
        "ok": True,
        "results": results,
        "elapsed": float(elapsed),
        "raw": raw_out,
        "model": MODEL_MAP[model_choice]["algo"],
        "preprocessed": bool(use_preprocess),
        "processed_image": processed_preview,
        "error": None,
    }


# ── Gradio Blocks ───────────────────────────────────────────────────────────
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Vietnamese Handwriting OCR — API", analytics_enabled=False) as demo:
        gr.Markdown(
            "### Vietnamese Handwriting OCR — Gradio API\n"
            "This Gradio surface exists so the React frontend can call `/gradio` via "
            "`@gradio/client`. The actual UI lives in the React SPA."
        )
        image_in = gr.Image(type="pil", label="Image")
        model_in = gr.Radio(
            choices=list(MODEL_MAP.keys()),
            value=list(MODEL_MAP.keys())[0],
            label="Model",
        )
        prep_in = gr.Checkbox(value=False, label="Adaptive preprocess")
        out = gr.JSON(label="Result")
        btn = gr.Button("Infer")
        btn.click(
            fn=infer,
            inputs=[image_in, model_in, prep_in],
            outputs=out,
            api_name="infer",
        )

        spell_in = gr.Textbox(visible=False)
        spell_out = gr.Textbox(visible=False)
        spell_btn = gr.Button(visible=False)
        spell_btn.click(
            fn=spell_correct,
            inputs=[spell_in],
            outputs=spell_out,
            api_name="spell_check",
        )
    return demo


# ── FastAPI assembly ────────────────────────────────────────────────────────
def build_app() -> FastAPI:
    app = FastAPI(title="Vietnamese Handwriting OCR")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "models": list(MODEL_MAP.keys()),
            "spa_mounted": FRONTEND_DIST.is_dir(),
        }

    @app.get("/api/models")
    def models() -> JSONResponse:
        return JSONResponse(
            {
                "models": [
                    {"key": k, "algo": v["algo"], "shape": v["shape"]}
                    for k, v in MODEL_MAP.items()
                ]
            }
        )

    demo = build_demo()
    app = gr.mount_gradio_app(app, demo, path="/gradio")

    # SPA mount comes last so it doesn't shadow /gradio, /healthz, /api/*.
    if FRONTEND_DIST.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(FRONTEND_DIST), html=True),
            name="spa",
        )
    else:
        @app.get("/")
        def _root() -> dict[str, Any]:
            return {
                "ok": True,
                "mode": "api-only",
                "hint": (
                    "Frontend dist not found at "
                    f"{FRONTEND_DIST}. Build with `pnpm build` inside viz/frontend "
                    "to enable Mode B, or run `pnpm dev` separately for Mode A."
                ),
                "gradio": "/gradio",
            }

    return app


# ── Optional Gradio share tunnel (Mode B) ───────────────────────────────────
def _maybe_open_tunnel(host: str, port: int) -> str | None:
    """Open a *.gradio.live tunnel that fronts the *whole* FastAPI app.

    Gradio's `setup_tunnel` exposes a local port; since both the SPA (/) and
    the Gradio API (/gradio) live behind that port, a single share URL is
    enough to serve everything.
    """
    if not SHARE:
        return None
    try:
        from gradio import networking  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        print(f"[viz] Could not import gradio.networking for share tunnel: {e}",
              file=sys.stderr)
        return None

    share_token = secrets.token_urlsafe(32)
    tunnel_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host

    # `setup_tunnel`'s signature has shifted across Gradio versions; try the
    # known shapes in order. All variants return the public URL string.
    last_err: Exception | None = None
    for args in (
        (tunnel_host, port, share_token, None, None),
        (tunnel_host, port, share_token, None),
        (tunnel_host, port, share_token),
    ):
        try:
            return networking.setup_tunnel(*args)  # type: ignore[arg-type]
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            print(f"[viz] Tunnel setup failed: {e}", file=sys.stderr)
            return None
    print(f"[viz] Could not call setup_tunnel (last error: {last_err})", file=sys.stderr)
    return None


def main() -> None:
    import uvicorn

    app = build_app()
    share_url = _maybe_open_tunnel(HOST, PORT)

    print("=" * 60)
    print(" Vietnamese Handwriting OCR — viz/")
    print("=" * 60)
    print(f"  Local API     : http://{HOST}:{PORT}/gradio")
    print(f"  Health check  : http://{HOST}:{PORT}/healthz")
    if FRONTEND_DIST.is_dir():
        print(f"  SPA mounted   : http://{HOST}:{PORT}/  (from {FRONTEND_DIST})")
    else:
        print("  SPA mounted   : no  (frontend/dist missing — Mode A)")
    if share_url:
        print(f"  Public tunnel : {share_url}")
        print(f"  SPA (public)  : {share_url}/")
        print(f"  Gradio (pub.) : {share_url}/gradio")
    print("=" * 60, flush=True)

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
