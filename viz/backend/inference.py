"""Inference primitives for the Vietnamese Handwriting OCR pipeline.

The five core functions below — MODEL_MAP, adaptive_crop_text_region,
adaptive_preprocess_for_ocr, parse_infer_output, run_inference — are copied
verbatim from ../../app.py to guarantee byte-for-byte reproducibility against
the existing Streamlit demo. Do not refactor them here; fix in app.py first.
"""

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

MODEL_MAP = {
    "SVTR (High Accuracy)": {
        "ckpt": SVTR_CKPT, "cfg": SVTR_CFG, "shape": "3,48,800", "algo": "SVTR",
    },
    "CRNN (High Speed)": {
        "ckpt": CRNN_CKPT, "cfg": CRNN_CFG, "shape": "3,32,640", "algo": "CRNN",
    },
}

# Image Preprocessing Helper
def adaptive_crop_text_region(img_cv: np.ndarray, base_pad_ratio: float = 0.15, trim_ratio: float = 0.02) -> np.ndarray:
    img_h, img_w = img_cv.shape[:2]

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Gọt bớt (trim) viền ảnh trước khi lấy bounding box để xoá nhiễu camera
    trim_x = int(img_w * trim_ratio)
    trim_y = int(img_h * trim_ratio)
    if trim_y > 0 and trim_x > 0:
        thresh_clean[0:trim_y, :] = 0
        thresh_clean[-trim_y:, :] = 0
        thresh_clean[:, 0:trim_x] = 0
        thresh_clean[:, -trim_x:] = 0

    coords = cv2.findNonZero(thresh_clean)
    if coords is None: return img_cv

    x, y, w, h = cv2.boundingRect(coords)

    # Thuật toán padding đồng bộ dựa hoàn toàn vào chiều cao (Height-based Strategy)
    pad_y = int(h * base_pad_ratio)
    pad_x = int(h * base_pad_ratio)

    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)
    return img_cv[y1:y2, x1:x2]

def adaptive_preprocess_for_ocr(img_pil: Image.Image) -> Image.Image:
    img_cv = np.array(img_pil)
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
    elif len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # 1. Chuyển sang Grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 2. Denoise TRƯỚC khi xử lý tương phản
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 3. Illumination Normalization (Khử nền loang lổ)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, bg_kernel)

    diff = cv2.absdiff(background, gray)
    normalized = 255 - diff

    # 4. Tăng tương phản nhẹ nhàng
    result = cv2.normalize(normalized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Định dạng lại BGR giả để tương thích API của hàm crop
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # 5. Cắt vùng chữ (Lúc này background đã trắng sạch 100%, OTSU sẽ cắt chính xác nét chữ đen)
    cropped_bgr = adaptive_crop_text_region(result_bgr, base_pad_ratio=0.15)

    return Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))

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
