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

## 🚀 Hướng dẫn sử dụng

> 💡 **Bạn KHÔNG cần cài đặt gì trên máy tính.** Toàn bộ notebook đều chạy trên **Kaggle** (miễn phí) và **tự động cài mọi thư viện cần thiết**. Bạn chỉ cần một tài khoản Kaggle và trình duyệt web.

### Mỗi notebook dùng để làm gì?

| Notebook | Công dụng |
| --- | --- |
| **`demo-viz-ui.ipynb`** | **Chạy web demo** — tải ảnh chữ viết tay lên và xem mô hình nhận diện. 👉 Người mới bắt đầu từ đây. |
| `crnn-uit-handwritten.ipynb` | Huấn luyện lại mô hình **CRNN** từ đầu. |
| `svtr-uit-handwitten.ipynb` | Huấn luyện lại mô hình **SVTR** từ đầu. |
| `eda-uit-handwriten.ipynb` | Phân tích, thống kê bộ dữ liệu UIT-HWDB. |

### ▶️ Cách 1 — Chạy web demo (khuyên dùng cho người mới)

Làm theo từng bước, **không cần cài bất cứ thứ gì**:

1. Đăng nhập [kaggle.com](https://www.kaggle.com) — tạo tài khoản miễn phí nếu chưa có.
2. Bấm **Create → New Notebook**. Vào menu **File → Import Notebook**, chọn tab **GitHub** và dán link, hoặc tải lên trực tiếp file `demo-viz-ui.ipynb`:
   `https://github.com/KhoaLeDang2375/Recognizing-Vietnamese-handwriting/blob/main/demo-viz-ui.ipynb`
3. Mở khung **Settings** (bên phải) và thiết lập:
   - **Accelerator:** chọn `GPU`
   - **Internet:** bật `On`
4. Bấm **Add Input**, thêm bộ dữ liệu UIT-HWDB và 2 model đã huấn luyện sẵn:
   [SVTR model](https://www.kaggle.com/models/thoandanh/svtr-vietnamese-handwriten) · [CRNN model](https://www.kaggle.com/models/thoandanh/crnn-vietnamese-handwriten)
5. Bấm **Run All** và chờ khoảng **3–5 phút** — notebook sẽ tự cài thư viện và build giao diện.
6. Kéo xuống cell cuối, copy đường link có dạng `https://xxxx.gradio.live`.
7. Mở link đó trên trình duyệt → **dùng demo ngay**. Link còn hoạt động khi notebook còn chạy, và có thể gửi cho người khác cùng xem.

### 🔁 Cách 2 — Huấn luyện lại mô hình

Mở `crnn-uit-handwritten.ipynb` hoặc `svtr-uit-handwitten.ipynb` trên Kaggle, bấm **Add Input** để đính kèm dataset UIT-HWDB, bật **GPU** trong Settings, rồi bấm **Run All**. Notebook tự cài thư viện và lưu checkpoint mô hình sau khi huấn luyện.

### 💻 Cách 3 — Chạy ở máy cá nhân *(nâng cao)*

Phần này dành cho người đã quen môi trường lập trình. Yêu cầu tự cài **PaddlePaddle**, **PaddleOCR**, **Node.js 18+** và có sẵn file model weights.

```bash
# 1) Backend — Gradio API (cổng 7860)
export PADDLEOCR_DIR=./PaddleOCR
export DICT_PATH=./vietnamses_dict.txt
export SVTR_CKPT=/đường/dẫn/svtr/best_accuracy
export CRNN_CKPT=/đường/dẫn/crnn/best_accuracy
export SVTR_CFG=/đường/dẫn/rec_svtr_stage2.yml
export CRNN_CFG=/đường/dẫn/rec_crnn_stage2.yml
python viz/backend/server.py

# 2) Frontend — mở một cửa sổ terminal khác
cd viz/frontend
npm install
npm run dev          # mở http://localhost:5173
```

Để tạo sẵn một link công khai `*.gradio.live` (giống chế độ chạy trên Kaggle):

```bash
cd viz/frontend && npm install && npm run build && cd ../..
GRADIO_SHARE=1 python viz/backend/server.py
```

## 🛠 Công nghệ sử dụng

PaddleOCR · PaddlePaddle · OpenCV · FastAPI · Gradio · React · Vite · TypeScript · Tailwind CSS

---

Đồ án phục vụ mục đích học tập — môn Tư duy tính toán cho Khoa học Dữ liệu (DS107).
