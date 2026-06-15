# ✍️ Nhận diện chữ viết tay tiếng Việt

> Đồ án môn **Tư duy tính toán cho Khoa học Dữ liệu (DS107)** — Trường Đại học Công nghệ Thông tin, ĐHQG-HCM.

Xây dựng và so sánh hai mô hình OCR nhận diện chữ viết tay tiếng Việt trên bộ dữ liệu **UIT-HWDB**, kèm một web demo trực quan.

##  Thành viên nhóm

| Họ và tên | MSSV |
| --- | --- |
| Lê Đăng Khoa | 23520740 |
| Lại Thị Thu Hương | 23520585 |
| Phan Trần Văn Khang | 23520708 |
| Trần Thị Kim Anh | 23520079 |

##  Tổng quan đề tài

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

##  Cấu trúc dự án

```text
├── crnn-uit-handwritten.ipynb   # Notebook huấn luyện CRNN (2-stage)
├── svtr-uit-handwitten.ipynb    # Notebook huấn luyện SVTR (2-stage)
├── eda-uit-handwriten.ipynb     # Notebook phân tích dữ liệu (EDA)
├── demo-viz-ui.ipynb            # Notebook chạy web demo (viz) trên Kaggle
├── demo-streamlit-ui.ipynb      # Notebook demo Streamlit cũ (backup)
├── app.py                       # App demo Streamlit cũ (bản tham chiếu)
├── rec_svtr_stage2.yml          # Config inference SVTR
├── rec_crnn_stage2.yml          # Config inference CRNN
├── vietnamses_dict.txt          # Bảng mã 161 ký tự tiếng Việt
└── viz/                         # Web demo (React + Gradio)
    ├── backend/                 # FastAPI + Gradio — dùng lại pipeline OCR
    └── frontend/                # React + Vite + Tailwind — trang báo cáo + demo
```

##  Hướng dẫn sử dụng

| Notebook | Mục đích |
| --- | --- |
| `demo-viz-ui.ipynb` | Chạy web demo nhận diện chữ viết tay. |
| `crnn-uit-handwritten.ipynb` | Huấn luyện mô hình CRNN. |
| `svtr-uit-handwitten.ipynb` | Huấn luyện mô hình SVTR. |
| `eda-uit-handwriten.ipynb` | Phân tích bộ dữ liệu UIT-HWDB. |

### Cách 1 — Chạy demo trên Kaggle *(khuyên dùng)*

Notebook demo đã được publish sẵn trên Kaggle, kèm dataset và pre-trained model:

** https://www.kaggle.com/code/khangphantrnvn/demo-ds107-viz**

Mở link → bấm **Copy & Edit** → bật **GPU** và **Internet** trong Settings → **Run All**. Notebook tự cài toàn bộ môi trường; cell cuối in ra đường link `https://<id>.gradio.live` — mở link đó để dùng demo. Link hoạt động khi notebook còn chạy và có thể chia sẻ cho người khác.

### Cách 2 — Huấn luyện mô hình

Chạy `crnn-uit-handwritten.ipynb` hoặc `svtr-uit-handwitten.ipynb` trên Kaggle với **GPU** và dataset UIT-HWDB đính kèm. Notebook tự cài thư viện và lưu checkpoint sau huấn luyện.

### Cách 3 — Chạy demo ở máy cá nhân

Yêu cầu: Python 3.10+, Node.js 18+, khuyến nghị GPU NVIDIA (CUDA). Các lệnh chạy từ thư mục gốc của repo.

**1. Tải mã nguồn và cài thư viện**

```bash
git clone https://github.com/KhoaLeDang2375/Recognizing-Vietnamese-handwriting.git
cd Recognizing-Vietnamese-handwriting
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git

pip install paddlepaddle-gpu                 # bản CPU: pip install paddlepaddle
pip install -r PaddleOCR/requirements.txt
pip install -r viz/backend/requirements.txt
```

**2. Tải pre-trained model về máy**

Hai mô hình đã huấn luyện được đăng trên Kaggle Models:

- **SVTR** — https://www.kaggle.com/models/thoandanh/svtr-vietnamese-handwriten
- **CRNN** — https://www.kaggle.com/models/thoandanh/crnn-vietnamese-handwriten

Tải qua giao diện web: mở trang model → chọn version (framework `pytorch`, instance `default`) → bấm **Download** → giải nén.

Hoặc dùng Kaggle CLI (cần file token `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle models instances versions download thoandanh/svtr-vietnamese-handwriten/pytorch/default/1
kaggle models instances versions download thoandanh/crnn-vietnamese-handwriten/pytorch/default/1
```

Sau khi giải nén, mỗi model có một thư mục checkpoint chứa `best_accuracy.pdparams`. Ghi nhớ đường dẫn checkpoint — là đường dẫn **không kèm đuôi `.pdparams`**:

- SVTR: `…/SVTR/Stage2/best_accuracy/best_accuracy`
- CRNN: `…/CRNN/Stage2/best_accuracy`

**3. Chạy demo**

```bash
# Backend — Gradio API ở cổng 7860
export PADDLEOCR_DIR=$(pwd)/PaddleOCR
export DICT_PATH=$(pwd)/vietnamses_dict.txt
export SVTR_CFG=$(pwd)/rec_svtr_stage2.yml
export CRNN_CFG=$(pwd)/rec_crnn_stage2.yml
export SVTR_CKPT=/duong/dan/toi/SVTR/Stage2/best_accuracy/best_accuracy
export CRNN_CKPT=/duong/dan/toi/CRNN/Stage2/best_accuracy
python viz/backend/server.py

# Frontend — mở một terminal khác
cd viz/frontend && npm install && npm run dev      # http://localhost:5173
```

> Muốn có link công khai `*.gradio.live` thay cho localhost: build frontend bằng `npm run build` trong `viz/frontend`, rồi chạy backend với `GRADIO_SHARE=1 python viz/backend/server.py`.

##  Công nghệ sử dụng

PaddleOCR · PaddlePaddle · OpenCV · FastAPI · Gradio · React · Vite · TypeScript · Tailwind CSS

---

Đồ án phục vụ mục đích học tập — môn Tư duy tính toán cho Khoa học Dữ liệu (DS107).
