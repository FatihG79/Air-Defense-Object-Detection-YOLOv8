# Air Defense Object Detection System using YOLOv8 🚀

This repository contains the source code and implementation details for a specialized computer vision pipeline designed for modern air defense and autonomous surveillance scenarios. The model is trained to detect, track, and classify four types of aerial targets with high precision.

---

## 📊 Project Quick Links
* **Live Kaggle Dataset:** [Kaggle Dataset](https://www.kaggle.com/datasets/caferfatihgltekin/air-defense-object-detection-dataset-yolov8)
* **Pre-trained Model Weights (Hugging Face):** [Hugging Face Model](https://huggingface.co/FatihG79/yolov8-air-defense-detection)
* **Interactive Kaggle Notebook:** [Kaggle Notebook](https://www.kaggle.com/code/caferfatihgltekin/yolov8-air-defense-inference-test)

---

## 🎯 Target Classes
The custom model is robustly trained to detect and classify **4 distinct classes**:
1. `f16` - Fighter jets and military aircraft.
2. `helicopter` - Civilian and military helicopters.
3. `quadcopter` - Multi-rotor drones and tactical UAVs.
4. `rocket` - Airborne rockets and high-speed missile threats.

---

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Framework:** Ultralytics YOLOv8 (Object Detection)
* **Data Curation:** Fabsketch 3D Simulation Environment (Synthetic Data Generation)
* **Annotation & Verification:** FiftyOne (Open-source Computer Vision Tool)
* **Development Environments:** Google Colab & Kaggle Notebooks (GPU accelerated inference)

---

## 📁 Dataset Structure & Curation
The dataset leverages a hybrid approach combining high-fidelity synthetic data and curated real-world imagery:
* **Synthetic Data:** Custom screenshots captured from the *Fabsketch* simulation software to model tactical target profiles under diverse viewing angles, altitudes, and lighting conditions.
* **Real-World Data:** Open-source internet images to maximize model generalization.
* **Labeling:** Manually annotated and verified using *FiftyOne* into standard YOLO bounding-box formats (`images/labels` split).

---

## 🚀 How to Run Inference

You can run immediate inference using the pre-trained weights (`model.pt`) included in the Kaggle dataset.

```python
from ultralytics import YOLO

# Load the custom trained model weights
model = YOLO('path_to/model.pt')

# Run inference on test images
results = model.predict(source='path_to/test_images', save=True, imgsz=640)
