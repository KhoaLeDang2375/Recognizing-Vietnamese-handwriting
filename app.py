import streamlit as st
import subprocess, sys, os, re, time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Configuration Constants
PADDLEOCR_DIR = os.environ.get('PADDLEOCR_DIR', '/kaggle/working/PaddleOCR')
WORK_DIR      = os.environ.get('WORK_DIR', '/kaggle/working')
DICT_PATH     = os.environ.get('DICT_PATH', '/kaggle/working/vietnamese_dict.txt')
SVTR_CKPT     = os.environ.get('SVTR_CKPT', '/kaggle/input/models/thoandanh/svtr-vietnamese-handwriten/pytorch/default/1/SVTR/Stage2/best_accuracy/best_accuracy')
CRNN_CKPT     = os.environ.get('CRNN_CKPT', '/kaggle/input/models/thoandanh/crnn-vietnamese-handwriten/pytorch/default/1/CRNN/Stage2/best_accuracy')
SVTR_CFG      = os.environ.get('SVTR_CFG', '/kaggle/working/rec_svtr_stage2.yml')
CRNN_CFG      = os.environ.get('CRNN_CFG', '/kaggle/working/rec_crnn_stage2.yml')
TEMP_DIR      = os.environ.get('TEMP_DIR', '/kaggle/working/temp_infer')

# Page Configuration
st.set_page_config(
    page_title="Nhận Dạng Chữ Viết Tay",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }

    /* Standardize Material Icons */
    .material-symbols-outlined {
        vertical-align: middle;
        font-size: 1.2rem;
        margin-right: 6px;
    }

    /* Main Header */
    .main-header {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header .icon-large {
        font-size: 2rem;
        color: #475569;
        margin: 0;
    }
    .main-header-text h1 { 
        font-size: 1.5rem; 
        font-weight: 600; 
        margin: 0; 
        color: #1e293b;
    }
    .main-header-text p { 
        font-size: 0.875rem; 
        color: #64748b; 
        margin: 0.25rem 0 0; 
    }

    /* Result Cards */
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .result-text {
        font-size: 1.125rem;
        font-weight: 500;
        color: #1e293b;
        line-height: 1.5;
        word-break: break-word;
    }
    
    /* Status Indicators */
    .meta-row {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.875rem;
        color: #64748b;
    }
    .status-dot {
        height: 8px;
        width: 8px;
        background-color: #cbd5e1;
        border-radius: 50%;
        display: inline-block;
    }
    .status-high { background-color: #10b981; }
    .status-medium { background-color: #f59e0b; }
    .status-low { background-color: #ef4444; }

    .model-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .info-box {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-size: 0.875rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        max-height: 300px;
        object-fit: contain;
    }
</style>
""", unsafe_allow_html=True)

MODEL_MAP = {
    "SVTR (High Accuracy)": {
        "ckpt": SVTR_CKPT, "cfg": SVTR_CFG, "shape": "3,48,800", "algo": "SVTR",
    },
    "CRNN (High Speed)": {
        "ckpt": CRNN_CKPT, "cfg": CRNN_CFG, "shape": "3,32,640", "algo": "CRNN",
    },
}

# Image Preprocessing Helper
def crop_text_region(img_cv: np.ndarray, padding: int = 10) -> np.ndarray:
    """Auto-crop the text region to remove extra background padding."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find dark regions (text) on light background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find all text contours
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img_cv  # Return original if no contours found
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Apply padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_cv.shape[1], x + w + padding)
    y2 = min(img_cv.shape[0], y + h + padding)
    
    return img_cv[y1:y2, x1:x2]

def advanced_preprocess_for_ocr(img_pil: Image.Image) -> Image.Image:
    img_cv = np.array(img_pil)

    if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
    elif len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    elif len(img_cv.shape) == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

    # Crop text region
    img_cv = crop_text_region(img_cv, padding=15)

    # CLAHE on L channel
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_cv = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Slight denoising
    img_cv = cv2.bilateralFilter(img_cv, d=5, sigmaColor=30, sigmaSpace=30)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# Inference Output Parsing Helper
def parse_infer_output(raw: str) -> list[dict]:
    results = []
    for line in raw.splitlines():
        m = re.search(r"result:\s+(.*?)\t([0-9]+\.[0-9]+)", line)
        if not m:
            m = re.search(r"Predicts of.*?:.*?'(.*?)'.*?([0-9]+\.[0-9]+)", line)
        if not m:
            m = re.search(r"\t\['(.*?)'.*?([0-9]+\.[0-9]+)", line)
            
        if m:
            text = m.group(1).strip()
            conf = float(m.group(2))
            if text:
                results.append({"text": text, "conf": conf})
    return results

# Inference Process Wrapper
def run_inference(img_path: str, model_key: str) -> tuple[list, float, str]:
    model = MODEL_MAP[model_key]
    cmd = [
        sys.executable,
        "tools/infer_rec.py",
        "-c", model["cfg"],
        "-o",
        f"Global.pretrained_model={model['ckpt']}",
        f"Global.infer_img={img_path}",
        f"Global.character_dict_path={DICT_PATH}",
        f"Global.use_space_char=True",
        f"Global.rec_image_shape={model['shape']}",
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=PADDLEOCR_DIR, timeout=60
    )
    elapsed = time.time() - t0
    raw = proc.stdout + proc.stderr
    results = parse_infer_output(raw)
    return results, elapsed, raw

# UI Layout Construction
st.markdown("""
<div class="main-header">
    <span class="material-symbols-outlined icon-large">document_scanner</span>
    <div class="main-header-text">
        <h1>Vietnamese Handwriting OCR</h1>
        <p>UIT Dataset System &middot; PaddleOCR Engine &middot; DS107</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    model_choice = st.radio(
        "Select Architecture",
        list(MODEL_MAP.keys()),
        index=0,
        help="SVTR: Transformer-based, higher accuracy.\nCRNN: CNN+RNN, faster inference."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🖼️ Image Processing")
    use_preprocess = st.toggle(
        "Adaptive Thresholding", 
        value=False,
        help="Applies OpenCV preprocessing to remove shadows and enhance dark inputs."
    )

    st.divider()
    st.markdown("### ℹ️ Model Metadata")
    info = MODEL_MAP[model_choice]
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | Architecture | `{info['algo']}` |
    | Tensor Shape | `{info['shape']}` |
    """)

    st.divider()
    show_raw = st.toggle("Display raw logs", value=False)
    st.caption("DS107 · Production Build · 2026")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Data Input")
    uploaded = st.file_uploader(
        "Upload handwriting image (Line-level)",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed"
    )

    if uploaded:
        img_raw = Image.open(uploaded).convert("RGB")
        
        if use_preprocess:
            img_display = advanced_preprocess_for_ocr(img_raw)
            cap_text = f"{uploaded.name} (Preprocessed)"
        else:
            img_display = img_raw
            cap_text = f"{uploaded.name} (Original)"

        st.image(img_display, caption=cap_text, use_column_width=True)

        with st.expander("Image Metadata"):
            st.write(f"**Dimensions:** {img_raw.size[0]} × {img_raw.size[1]} px")
            st.write(f"**Aspect Ratio:** {img_raw.size[0]/img_raw.size[1]:.2f}")
    else:
        st.markdown("""
        <div class="info-box">
            <span class="material-symbols-outlined">lightbulb</span>
            <b>Usage Instructions</b><br><br>
            Upload a line-level image (one text line per image). If the image contains uneven lighting or shadows, enable <b>Adaptive Thresholding</b> in the sidebar for optimal results.
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 Inference Results")

    if uploaded:
        os.makedirs(TEMP_DIR, exist_ok=True)
        tmp_path = os.path.join(TEMP_DIR, "temp_infer.jpg")
        img_display.save(tmp_path, format="JPEG")

        algo = MODEL_MAP[model_choice]["algo"]
        st.markdown(f"""
        <div class="model-badge">
            <span class="material-symbols-outlined" style="font-size: 1rem; margin-right: 4px;">memory</span> 
            Processing via {algo}
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Executing inference..."):
            try:
                results, elapsed, raw_out = run_inference(tmp_path, model_choice)
            except subprocess.TimeoutExpired:
                st.error("Inference timeout. The model took too long to respond.")
                results, elapsed, raw_out = [], 0, ""
            except Exception as e:
                st.error(f"System error: {e}")
                results, elapsed, raw_out = [], 0, str(e)

        if results:
            for r in results:
                text = r["text"]
                conf = r["conf"]

                if conf >= 0.85:
                    status_class = "status-high"
                elif conf >= 0.6:
                    status_class = "status-medium"
                else:
                    status_class = "status-low"

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-text">{text}</div>
                    <div class="meta-row">
                        <span class="status-dot {status_class}"></span>
                        <span>Confidence: {conf:.1%}</span>
                        <span style="margin-left: auto; font-size: 0.75rem; color: #94a3b8;">
                            <span class="material-symbols-outlined" style="font-size: 0.875rem; margin-right: 2px;">timer</span>
                            {elapsed:.2f}s
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("No text detected. Try enabling Adaptive Thresholding.")
            if show_raw:
                st.code(raw_out[-2000:] if raw_out else "(empty)", language="text")

        if show_raw and raw_out:
            st.markdown("### 📋 System Logs")
            st.code(raw_out[-3000:], language="text")

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem; color: #cbd5e1; border: 1px dashed #e2e8f0; border-radius: 8px;">
            <span class="material-symbols-outlined" style="font-size: 3rem; margin: 0;">inbox</span>
            <p style="font-size:0.875rem; margin-top:1rem; color: #64748b;">Awaiting data input</p>
        </div>
        """, unsafe_allow_html=True)
