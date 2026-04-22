# ✍️ Vietnamese Handwriting OCR

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.0.0-deepskyblue?logo=paddlepaddle&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Overview

This project provides a robust Machine Learning solution for **Vietnamese Handwriting Recognition**. Built on top of the powerful **PaddleOCR** ecosystem, it achieves high-accuracy character sequence recognition targeted at the UIT-HWDB (Vietnamese Handwritten Database). 

It specifically trains and fine-tunes two state-of-the-art architectures:
- **SVTR (Single Visual model for Text Recognition):** A Transformer-based architecture prioritizing ultimate accuracy.
- **CRNN (Convolutional Recurrent Neural Network):** A CNN-RNN hybrid for high-speed, lightweight inference.

The project includes an end-to-end pipeline from a custom 2-stage training strategy (Warmup & Full Fine-tuning) on Kaggle Notebooks to a production-ready **Streamlit web application** for interactive real-time inference.

## ✨ Features

* **Dual Architecture Support:** Switch seamlessly between SVTR (Accuracy) and CRNN (Speed).
* **2-Stage Fine-Tuning Strategy:** Employs CTC Head warmup followed by full backbone fine-tuning tailored for Vietnamese text constraints.
* **Custom Data Augmentation:** Built-in `HandwritingAug` pipeline designed explicitly to prevent overfitting on handwritten strokes.
* **Interactive UI:** A highly polished Streamlit web app with dynamic OpenCV-based adaptive thresholding and auto-cropping for degraded images.
* **Kaggle-Ready Deployment:** Specialized notebooks to train and expose the Streamlit UI directly from Kaggle via `ngrok`.

## 🏗 Project Structure

```text
├── CRNN/                            # CRNN output logs & inference results
├── SVTR/                            # SVTR output logs & inference results
├── test_img/                        # Sample handwritten images for demonstration
├── app.py                           # Main Streamlit web application
├── crnn-uit-handwritten.ipynb       # Jupyter Notebook for Training CRNN (2-stage)
├── svtr-uit-handwitten.ipynb        # Jupyter Notebook for Training SVTR (2-stage)
├── demo-streamlit-ui.ipynb          # Deployment notebook integrating ngrok + Streamlit
├── vietnamses_dict.txt              # Unified 161-character Vietnamese dictionary 
└── README.md                        # Project documentation
```

## ⚙️ Installation

### 1. Requirements

Ensure you have Python 3.8+ installed. This project relies on `paddlepaddle-gpu` (or standard `paddlepaddle` for CPU) and `PaddleOCR`.

```bash
# Clone the repository
git clone https://github.com/KhoaLeDang2375/Recognizing-Vietnamese-handwriting.git
cd Recognizing-Vietnamese-handwriting

# Clone PaddleOCR locally
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
cd ..
```

### 2. Python Dependencies

```bash
pip install streamlit pyngrok Pillow opencv-python-headless
```

### 3. Setup Model Weights

Download the exported `.pdparams` models from the Kaggle experiments for SVTR/CRNN or train them using the provided notebooks:
- [SVTR Vietnamese Handwriting Model](https://www.kaggle.com/models/thoandanh/svtr-vietnamese-handwriten)
- [CRNN Vietnamese Handwriting Model](https://www.kaggle.com/models/thoandanh/crnn-vietnamese-handwriten)

## 🚀 Usage

### Running the Web Application (Streamlit)

The UI allows for uploading line-level handwriting images and processing them instantly.

```bash
# Set environment variables pointing to your downloaded model weights & configs
export PADDLEOCR_DIR="./PaddleOCR"
export DICT_PATH="./vietnamses_dict.txt"
export SVTR_CKPT="/path/to/svtr/best_accuracy"
export CRNN_CKPT="/path/to/crnn/best_accuracy"
export SVTR_CFG="/path/to/rec_svtr_stage2.yml"
export CRNN_CFG="/path/to/rec_crnn_stage2.yml"

# Launch the app
streamlit run app.py
```

### Kaggle Cloud Deployment

To easily run the web application on a Kaggle GPU instance without local setup:
1. Open `demo-streamlit-ui.ipynb` in your Kaggle environment.
2. Attach your PaddleOCR model dataset.
3. Configure your `NGROK_TOKEN` within Kaggle Secrets.
4. Run all cells to receive a live public Ngrok URL!

## 🧪 Technical Details

### Inference Pre-Processing (OpenCV)
The Streamlit app applies a custom `adaptive_preprocess_for_ocr` sequence to handle imperfect environmental capture, particularly uneven illumination and shadows:
1. **Grayscale & Denoising:** Converts the image to grayscale and applies Fast Non-Local Means Denoising to remove noise before contrast adjustments.
2. **Illumination Normalization:** Uses morphological operations to estimate and remove uneven background illumination (shadows/blotchy backgrounds).
3. **Contrast Normalization:** Gently enhances contrast using Min-Max normalization.
4. **Adaptive Otsu's Binarization & Auto-Crop:** Trims image borders to remove camera artifacts, applies Otsu's thresholding to precisely isolate text, and uses a height-based padding strategy for consistent cropping.

## 📦 Dependencies

* `paddlepaddle-gpu` (v2.6+ / v3.0+)
* `paddleocr`
* `streamlit`
* `opencv-python-headless`
* `Pillow`
* `pyngrok` (for cloud UI hosting)

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you want to propose performance improvements, UI enhancements, or add new features like post-correction Spell Checkers.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
