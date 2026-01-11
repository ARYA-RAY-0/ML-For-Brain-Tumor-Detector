### 🧠 Brain Tumor Detector

AI-powered brain tumor detection from MRI scans using Convolutional Neural Networks (CNNs), Python, TensorFlow/Keras, OpenCV, and Gradio for an interactive demo.

---

## 📌 Project Overview

This project detects the presence of brain tumors from MRI scan images using a CNN-based image classification approach. The system processes MRI images, applies preprocessing techniques, and predicts whether a tumor is present.

An interactive Gradio interface allows users to upload MRI images and receive instant predictions.

---

## ✨ Key Features

* MRI image preprocessing and resizing with OpenCV
* CNN-based binary classification (Tumor / No Tumor)
* Interactive Gradio web interface
* End-to-end ML inference workflow
* Designed for use in Google Colab or local environments

---

## 🚀 Demo & Usage

The Gradio interface can be launched locally or in a Google Colab notebook.

Example usage:

```python
from tensorflow.keras.models import load_model
import gradio as gr
import cv2
import numpy as np

model = load_model("brain_tumor_detector.keras")

def predict(img):
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    prediction = model.predict(img_array)
    return "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

iface = gr.Interface(fn=predict, inputs="image", outputs="label")
iface.launch()
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ARYA-RAY-0/ML-For-Brain-Tumor-Detector.git
```

Install dependencies manually or via Colab:

```bash
pip install tensorflow opencv-python numpy gradio matplotlib
```

Run the notebook or demo script.

---

## 📦 Model & Dataset

* Trained model files (`.keras`) are **not included** in this repository due to size constraints.
* The notebook demonstrates the **complete preprocessing and inference pipeline**.
* The model can be retrained using publicly available MRI datasets such as:

  * Kaggle Brain MRI Images for Brain Tumor Detection (add link if you want)

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Gradio
* Matplotlib

---

## 🔮 Future Improvements

* Improve accuracy with additional training data
* Add Grad-CAM heatmaps for tumor localization
* Deploy as a public demo on Hugging Face Spaces

