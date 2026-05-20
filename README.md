# ✍️ Nhận diện chữ viết tay tiếng Việt

> Đồ án môn **Tư duy tính toán cho Khoa học Dữ liệu (DS107)** — Trường Đại học Công nghệ Thông tin, ĐHQG-HCM.

Xây dựng và so sánh hai mô hình OCR nhận diện chữ viết tay tiếng Việt trên bộ dữ liệu **UIT-HWDB**, kèm một web demo trực quan.

## 👥 Thành viên nhóm

| Họ và tên | MSSV |
| --- | --- |
| Lê Đăng Khoa | 23520740 |
| Lại Thị Thu Hương | 23520585 |
| Phan Trần Văn Khang | 23520708 |
| Trần Thị Kim Anh | 23520079 |

## 📌 Tổng quan đề tài

Đề tài giải quyết bài toán nhận diện (OCR) ảnh dòng chữ viết tay tiếng Việt thành văn bản số. Khó khăn đặc thù của tiếng Việt nằm ở hệ thống dấu thanh và dấu phụ dày đặc, nhiều ký tự dễ nhầm lẫn, cùng với nét chữ viết tay biến dạng và hiện tượng dính nét.

Nhóm triển khai và so sánh **hai kiến trúc** tiêu biểu trên engine **PaddleOCR**:

- **CRNN** — Backbone ResNet-34 + Neck BiLSTM (hidden 256) + Head CTC. Hướng tiếp cận kết hợp CNN với RNN tuần tự.
- **SVTR** — Backbone SVTRNet phân cấp 3 stage (Local/Global Mixer) + Neck SequenceEncoder + Head CTC. Hướng tiếp cận Transformer thuần thị giác.

Cả hai mô hình dùng chung pipeline xử lý và hàm mất mát CTC (bảng mã 161 lớp ký tự), được huấn luyện theo chiến lược **2-Stage Fine-Tuning**:

1. **Stage 1 — Warm-up:** đóng băng Backbone, chỉ huấn luyện Neck/Head để căn chỉnh với phân phối ký tự tiếng Việt.
2. **Stage 2 — Fine-tuning:** mở khóa toàn mạng, tinh chỉnh sâu với learning rate nhỏ.

### Dữ liệu

Bộ **UIT-HWDB** (UIT Vietnamese Handwritten Database) — dữ liệu chữ viết tay tiếng Việt dạng offline. Đồ án sử dụng tập con **UIT-HWDB-line** gồm 7.273 dòng văn bản; bảng mã ký tự gồm 161 lớp (160 ký tự tiếng Việt + 1 ký tự blank cho thuật toán CTC).

### Kết quả chính

| Mô hình | Word Accuracy | CER ↓ | WER ↓ | NED ↓ | Confidence |
| --- | :---: | :---: | :---: | :---: | :---: |
| CRNN | ≈ 0.00% | 37.40% | 86.80% | 0.374 | 61.3% |
| **SVTR** | **11.44%** | **9.50%** | **29.10%** | **0.095** | **92.7%** |

→ **SVTR** vượt trội trên toàn bộ chỉ số và là kiến trúc được nhóm lựa chọn.

## 🗂 Cấu trúc dự án

```text
├── crnn-uit-handwritten.ipynb   # Notebook huấn luyện CRNN (2-stage)
├── svtr-uit-handwitten.ipynb    # Notebook huấn luyện SVTR (2-stage)
├── eda-uit-handwriten.ipynb     # Notebook phân tích dữ liệu (EDA)
├── demo-viz-ui.ipynb            # Notebook chạy web demo (viz) trên Kaggle
├── demo-streamlit-ui.ipynb      # Notebook demo Streamlit cũ (backup)
├── app.py                       # App demo Streamlit cũ (bản tham chiếu)
├── vietnamses_dict.txt          # Bảng mã 161 ký tự tiếng Việt
└── viz/                         # Web demo (React + Gradio)
    ├── backend/                 # FastAPI + Gradio — dùng lại pipeline OCR
    └── frontend/                # React + Vite + Tailwind — trang báo cáo + demo
```

## 🚀 Cách chạy code

### 1. Huấn luyện mô hình (Kaggle, cần GPU)

Mở `crnn-uit-handwritten.ipynb` hoặc `svtr-uit-handwitten.ipynb` trên Kaggle, đính kèm dataset UIT-HWDB, bật GPU rồi chạy toàn bộ cell. Notebook `eda-uit-handwriten.ipynb` dùng để phân tích, thống kê bộ dữ liệu.

### 2. Chạy web demo trên Kaggle — *cách nhanh nhất*

`demo-viz-ui.ipynb` tự cài môi trường, build giao diện và tạo một URL công khai.

1. Tải `demo-viz-ui.ipynb` lên Kaggle (hoặc *File → Import Notebook*).
2. Bật **GPU** và **Internet** trong phần Settings.
3. **Add Input:** dataset UIT-HWDB và 2 model đã huấn luyện ([SVTR](https://www.kaggle.com/models/thoandanh/svtr-vietnamese-handwriten), [CRNN](https://www.kaggle.com/models/thoandanh/crnn-vietnamese-handwriten)).
4. **Run All** → notebook in ra URL `https://xxxx.gradio.live`, mở để dùng demo.

### 3. Chạy web demo ở máy local

Yêu cầu: đã cài PaddlePaddle + PaddleOCR và có sẵn model weights; Node.js 18+.

```bash
# 1) Backend — Gradio API (cổng 7860)
export PADDLEOCR_DIR=./PaddleOCR
export DICT_PATH=./vietnamses_dict.txt
export SVTR_CKPT=/đường/dẫn/svtr/best_accuracy
export CRNN_CKPT=/đường/dẫn/crnn/best_accuracy
export SVTR_CFG=/đường/dẫn/rec_svtr_stage2.yml
export CRNN_CFG=/đường/dẫn/rec_crnn_stage2.yml
python viz/backend/server.py

# 2) Frontend — mở terminal khác
cd viz/frontend
npm install
npm run dev          # http://localhost:5173
```

Hoặc chạy gộp trong một tiến trình, có sẵn link công khai `*.gradio.live`:

```bash
cd viz/frontend && npm install && npm run build && cd ../..
GRADIO_SHARE=1 python viz/backend/server.py
```

> Bản demo Streamlit cũ (`app.py` + `demo-streamlit-ui.ipynb`) được giữ lại làm backup, chạy bằng `streamlit run app.py`.

## 🛠 Công nghệ sử dụng

PaddleOCR · PaddlePaddle · OpenCV · FastAPI · Gradio · React · Vite · TypeScript · Tailwind CSS

---

Đồ án phục vụ mục đích học tập — môn Tư duy tính toán cho Khoa học Dữ liệu (DS107).
