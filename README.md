# PyCamAI - Real-Time AI Face Swap 🎭

PyCamAI is a highly optimized, real-time face-swapping application built with Python, OpenCV, and InsightFace. It is specifically engineered to run efficiently on standard CPUs by implementing asynchronous multithreading and resource management techniques.

## 🚀 Key Features

* **Real-Time Processing:** Swaps faces directly from the webcam feed.
* **Multithreading Architecture:** Separates the camera stream from the heavy AI inference, eliminating UI freezing and camera stuttering.
* **CPU Optimization:** Dynamically scales detection resolution (`det_size`) to ensure stable FPS on machines without dedicated GPUs.
* **Robust Resource Management:** Includes safe fallbacks and explicit error handling for missing models or assets.
* **Live Observability:** Built-in FPS counter and AI-detection state logging directly on the UI stream.

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **Computer Vision:** OpenCV (`cv2`)
* **AI Engine:** InsightFace (`buffalo_l` model)
* **Inference:** ONNX Runtime (CPU Execution Provider)

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/pycamai.git](https://github.com/YOUR_USERNAME/pycamai.git)
   cd pycamai

   python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate


# How use:
pip install -r requirements.txt

Resource Placement:

Place your target face image inside the assets/ folder and name it image.jpg.


Download the inswapper_128.onnx model and place it in your system's InsightFace directory: ~/.insightface/models/inswapper_128.onnx.


