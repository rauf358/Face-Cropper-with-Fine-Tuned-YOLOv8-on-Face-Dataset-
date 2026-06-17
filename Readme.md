Here is a professional, highly polished `README.md` file for your Face Cropper project. You can copy and paste this directly into your GitHub repository.

---

# 📸 Real-Time YOLOv8 Face Cropper

An ultra-low latency, high-throughput face detection and automated cropping pipeline. This project fine-tunes a **YOLOv8n** architecture on the complex **WIDER Face dataset** to deliver real-time face extraction optimized for surveillance feeds and downstream deep learning preprocessing.

## 🚀 Key Features

* **Ultra-Low Latency:** Achieves an inference time of just **1.9ms per image**, enabling real-time processing speeds exceeding **240+ FPS**.
* **High-Density Detection:** Trained to handle crowded, complex environments with heavy occlusion.
* **Automated Cropping Pipeline:** Instantly detects bounding boxes and crops faces for seamless integration into facial recognition or emotion analysis pipelines.
* **Optimized Architecture:** Configured for single-class object detection (`nc=1`) using Automatic Mixed Precision (AMP) for maximum GPU efficiency.

## 📊 Performance Benchmarks

*Evaluated on an NVIDIA Tesla T4 GPU (15GB VRAM) at 640x640 resolution.*

| Metric | Value |
| --- | --- |
| **Inference Latency** | 1.9 ms / image |
| **Throughput** | ~240+ FPS |
| **Precision (P)** | 77.3% |
| **mAP50** | 54.8% |
| **Recall (R)** | 47.3% |

*Note: These metrics reflect a 2-epoch baseline validation run demonstrating the high-throughput viability of the data ingestion pipeline.*

## 🛠️ Tech Stack & Environment

* **Frameworks:** PyTorch 2.6.0+cu124, Ultralytics YOLOv8 (8.3.159)
* **Language:** Python 3.11.13
* **Hardware:** CUDA 12.4 / NVIDIA Tesla T4
* **Dataset:** WIDER Face (12,880 Training Images, 3,225 Validation Images, 39,675 Instances)

## ⚙️ Installation

1. Clone the repository:
```bash

```



git clone https://github.com/rauf358/face-cropper.git
cd face-cropper

```

2. Create a virtual environment and activate it:
   ```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```

3. Install the required dependencies:
```bash

```



pip install ultralytics torch torchvision opencv-python

```

## 💻 Usage

### 1. Run Inference & Auto-Crop via Python
You can use the provided Python script to detect and save cropped faces from an image or video stream.

```python
from ultralytics import YOLO
import cv2
import os

# Load the fine-tuned model
model = YOLO('path/to/best.pt')

# Load image
img_path = 'sample.jpg'
image = cv2.imread(img_path)

# Run detection
results = model(image)

# Create directory for cropped faces
os.makedirs('cropped_faces', exist_ok=True)

# Extract and save crops
for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    cropped_face = image[y1:y2, x1:x2]
    cv2.imwrite(f'cropped_faces/face_{i}.jpg', cropped_face)

print(f"Successfully cropped and saved {len(results[0].boxes)} faces.")

```

### 2. Run via CLI

To test the model quickly on a video or webcam feed:

```bash
yolo task=detect mode=predict model=path/to/best.pt source=0 show=True save_crop=True

```

## 🧠 Training Details

The model was trained using the `yolov8n.pt` base model with the following key hyperparameter configurations:

* **Image Size:** 640x640
* **Batch Size:** 16
* **Optimizer:** AdamW (lr=0.002, momentum=0.9)
* **Augmentations:** RandAugment, Mosaic (1.0), Fliplr (0.5), Erasing (0.4)

## 🤝 Connect

If you found this project helpful or want to discuss Edge AI and Computer Vision, feel free to reach out!

* **LinkedIn:** [Abdul Rauf](https://www.google.com/search?q=https://www.linkedin.com/in/abdul-rauf-068b042b8)
