
# Brain Tumor Detector

**AI-powered brain tumor detection from MRI scans** using Convolutional Neural Networks (CNNs), Python, TensorFlow, OpenCV, and Gradio for an interactive demo.

## Project Overview

This project detects the presence of brain tumors from MRI scan images. It uses a CNN trained on labeled MRI images to classify whether a tumor is present. The model is deployed with a Gradio interface, allowing users to upload MRI images and get instant predictions.

Key features:

* Image preprocessing and resizing with OpenCV
* CNN-based brain tumor detection
* Interactive web demo using Gradio
* Ready-to-run Python scripts

## Demo

You can try the live demo by running the Gradio interface in your local environment or Colab notebook.

Example usage in Colab:

```python
!pip install -r requirements.txt
from tensorflow.keras.models import load_model
import gradio as gr

model = load_model("brain_tumor_detector.keras")  # Load your trained model

def predict(img):
    # Preprocess image
    import cv2
    import numpy as np
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    prediction = model.predict(img_array)
    return "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

iface = gr.Interface(fn=predict, inputs="image", outputs="label")
iface.launch(share=True)
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/ARYA-RAY-0/ML-For-Brain-Tumor-Detector.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo script or notebook.

## Requirements

```text
tensorflow>=2.12
opencv-python
numpy
gradio
matplotlib
```

## Dataset

MRI scan images for training and testing were obtained from [link to dataset] (optional: add Kaggle or public dataset link).

## Future Improvements

* Improve accuracy with more data
* Add heatmap visualization for tumor regions
* Deploy on Hugging Face Spaces for permanent demo


