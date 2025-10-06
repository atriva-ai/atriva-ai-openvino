
## **📝 README.md (Atriva AI API)**

```md
# Atriva AI API with OpenVINO 🚀

This is a FastAPI-based AI API that leverages **OpenVINO** for optimized deep learning inference.  
It provides a RESTful interface for running AI models, such as object detection and image classification.

## **📂 Project Structure**
```plaintext
atriva-ai-openvino/
│── app/
│   ├── routes.py         # API route definitions
│   ├── services.py       # AI model processing logic
│   ├── models.py         # Data models and schemas
│   ├── model_capabilities.py  # Model capabilities and metadata
│   ├── shared_data.py    # Shared data utilities
│── models/               # Pretrained OpenVINO models
│   ├── yolov8n/          # YOLOv8n object detection model
│   ├── lprnet/           # LPRNet license plate recognition model
│   └── vehicle_tracking/ # Vehicle detection and tracking model
│── tests/                # Comprehensive testing suite
│   ├── test_runner.py    # Main test runner
│   ├── test_yolov8n.py   # YOLOv8n specific tests
│   ├── test_lprnet.py    # LPRNet specific tests
│   ├── test_vehicle_tracking.py  # Vehicle tracking tests
│   └── setup.sh          # Test environment setup
│── main.py               # Entry point for FastAPI
│── config.py             # Configuration settings
│── requirements.txt      # Python dependencies
│── Dockerfile            # Docker configuration
│── README.md             # Project documentation
```

## **⚡ Features**
✅ FastAPI-based AI API  
✅ OpenVINO optimization for inference  
✅ Dockerized for easy deployment  
✅ Comprehensive testing suite  
✅ Multiple AI models supported:
   - YOLOv8n Object Detection
   - LPRNet License Plate Recognition
   - Vehicle Detection and Tracking  

## **🔧 Setup & Installation**

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/atriva-ai/atriva-ai-openvino.git
cd atriva-ai-openvino
```

### **2️⃣ Create a Virtual Environment**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### **3️⃣ Download AI Models**
```sh
# Download all required model files
cd tests
python test_runner.py --download-models

# Or download individual models
python test_runner.py --model yolov8n --download
python test_runner.py --model lprnet --download
python test_runner.py --model vehicle_tracking --download
```

**📝 Important**: Model binary files (.pt, .bin, .xml) are not included in the repository due to size constraints. They will be downloaded automatically when needed.

### **4️⃣ Run the API Locally**
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Access the API documentation at:  
👉 **http://localhost:8000/docs**

## **🐳 Running with Docker**
### **1️⃣ Build the Docker Image**
```sh
docker build -t atriva-ai-openvino .
```

### **2️⃣ Run the Container**
```sh
docker run -d -p 8000:8000 --name ai-openvino-container atriva-ai-openvino
```
Now, visit:  
👉 **http://localhost:8000/docs**

## **📥 Model Build Flow & Management**

### **🏗️ Model Build Architecture**

The AI service uses a **pre-built model approach** for optimal performance:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Development   │    │   Docker Build   │    │   Production    │
│   (Host)        │    │   (Container)    │    │   (Runtime)     │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Scripts       │    │ • Copy models    │    │ • Load models   │
│ • Model conv.   │───▶│ • Install deps   │───▶│ • Run inference │
│ • Testing       │    │ • Fast build     │    │ • API endpoints │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **📋 Requirements Structure**

| File | Purpose | Python Version | Dependencies |
|------|---------|----------------|-------------|
| `requirements.txt` | **Docker AI Service** | 3.12 | FastAPI, OpenVINO, NumPy, OpenCV |
| `tests/requirements.txt` | **Testing Environment** | **3.11** | Testing framework, utilities |
| `scripts/requirements.txt` | **Model Conversion** | **3.11** | PyTorch, Ultralytics, OpenVINO |

### **🚀 Quick Start (Recommended)**

**Models are pre-built and ready to use:**

```sh
# 1. Build Docker image (models already included)
docker build -t atriva-ai-openvino .

# 2. Run AI service
docker run -d -p 8001:8001 --name ai-inference atriva-ai-openvino

# 3. Test API
curl http://localhost:8001/models
```

### **🔧 Development Workflow**

#### **⚠️ Python Version Requirements**
- **Docker AI Service**: Python 3.12 (handled by Dockerfile)
- **Testing Environment**: **Python 3.11 required** (PyTorch compatibility)
- **Scripts Environment**: **Python 3.11 required** (PyTorch compatibility)

#### **For Testing & Development:**
```sh
# Setup test environment with Python 3.11
cd tests
pyenv local 3.11.13  # or use python3.11
python3.11 -m venv test-venv-py311
source test-venv-py311/bin/activate
pip install -r requirements.txt

# Run tests
python test_runner.py --model yolov8n --input test_images/sample.jpg
```

#### **For Model Conversion:**
```sh
# Setup scripts environment with Python 3.11
cd scripts
pyenv local 3.11.13  # or use python3.11
python3.11 -m venv scripts-venv-py311
source scripts-venv-py311/bin/activate
pip install -r requirements.txt

# Convert PyTorch to OpenVINO
python convert_to_openvino.py --size n
```

#### **Why Python 3.11?**
- **PyTorch compatibility**: PyTorch doesn't have pre-built wheels for Python 3.13
- **NumPy compatibility**: Avoids NumPy 2.x compatibility issues
- **Stable ML ecosystem**: Most ML libraries are tested with Python 3.11

### **📦 Model Management**

#### **Available Models:**
- ✅ **YOLOv8n** - Object detection (pre-built)
- ✅ **YOLOv8s** - Object detection (pre-built)  
- ✅ **YOLOv8m** - Object detection (pre-built)
- ✅ **LPRNet** - License plate recognition (pre-built)
- ✅ **Vehicle Tracking** - Vehicle detection (pre-built)

#### **Model Files Structure:**
```
models/
├── yolov8n/
│   ├── yolov8n.xml          # OpenVINO model structure
│   ├── yolov8n.bin          # OpenVINO model weights
│   └── model.json           # Model configuration
├── lprnet/
│   ├── lprnet.xml
│   ├── lprnet.bin
│   └── model.json
└── vehicle_tracking/
    ├── vehicle_tracking.xml
    ├── vehicle_tracking.bin
    └── model.json
```

#### **Verify Model Installation:**
```sh
# Check all models are present
ls models/*/model.json
ls models/*/*.xml
ls models/*/*.bin

# Expected output:
# models/lprnet/model.json models/vehicle_tracking/model.json models/yolov8n/model.json
# models/lprnet/lprnet.xml models/vehicle_tracking/vehicle_tracking.xml models/yolov8n/yolov8n.xml
# models/lprnet/lprnet.bin models/vehicle_tracking/vehicle_tracking.bin models/yolov8n/yolov8n.bin
```

### **🔄 Adding New Models**

1. **Create model directory:**
   ```sh
   mkdir models/new_model
   ```

2. **Add model files:**
   - `new_model.xml` - OpenVINO structure
   - `new_model.bin` - OpenVINO weights  
   - `model.json` - Configuration

3. **Update model capabilities:**
   ```python
   # Add to app/model_capabilities.py
   ```

4. **Test the model:**
   ```sh
   python test_runner.py --model new_model --input test_images/sample.jpg
   ```

## **🚨 Troubleshooting**

### **Python Version Issues**

#### **Error: "No matching distribution found for torch"**
```bash
# Problem: Using Python 3.13
# Solution: Use Python 3.11
pyenv local 3.11.13
python3.11 -m venv venv-py311
source venv-py311/bin/activate
pip install -r requirements.txt
```

#### **Error: "NumPy compatibility issues"**
```bash
# Problem: NumPy 2.x with PyTorch
# Solution: Downgrade NumPy
pip install "numpy<2"
```

#### **Error: "ModuleNotFoundError: No module named 'ultralytics'"**
```bash
# Problem: Missing PyTorch dependencies
# Solution: Use Python 3.11 environment
cd scripts
pyenv local 3.11.13
python3.11 -m venv scripts-venv-py311
source scripts-venv-py311/bin/activate
pip install -r requirements.txt
```

### **Model Issues**

#### **Error: "Unable to read the model: model.xml"**
```bash
# Problem: Corrupted model files (HTML instead of XML)
# Solution: Regenerate models using scripts
cd scripts
source scripts-venv-py311/bin/activate
python convert_to_openvino.py --size n
```

#### **Error: "Model file not found: yolov8n.pt"**
```bash
# Problem: Missing PyTorch model
# Solution: Download model first
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

#### **Error: "404 Not Found" when downloading models**
```bash
# Problem: Outdated model URLs
# Solution: Use working URLs
# Working PyTorch model URL:
# https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt

# Note: ONNX models are not pre-built, convert from PyTorch:
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx')
print('ONNX model exported')
"
```

### **Model File Structure**
Each model directory contains:
- `model.json` - Model configuration and metadata
- `metadata.yaml` - Additional model information
- `*.xml` - OpenVINO model structure (downloaded)
- `*.bin` - OpenVINO model weights (downloaded)
- `*.pt` - PyTorch model weights (optional, for conversion)

**Note**: Binary files (.pt, .bin, .xml) are not stored in Git due to size constraints. They are downloaded automatically when needed.

## **🛠 API Endpoints**
| Method | Endpoint         | Description          |
|--------|-----------------|----------------------|
| `GET`  | `/`             | Health check        |
| `POST` | `/predict`      | Run AI inference    |

## **🧪 Running Tests**

**⚠️ Prerequisites**: Make sure you have downloaded the required models first (see [Model Setup](#-model-setup--management) section above).

```sh
# Setup test environment
cd tests
./setup.sh
source venv/bin/activate

# Download models (if not already done)
python test_runner.py --download-models

# Run all tests
python test_runner.py --model all --input test_images/sample.jpg

# Test specific model
python test_yolov8n.py --input test_images/sample_cars.jpg

# Test with different model sizes
python test_yolov8_openvino.py --input test_images/sample_cars.jpg --size n  # nano
python test_yolov8_openvino.py --input test_images/sample_cars.jpg --size s  # small
python test_yolov8_openvino.py --input test_images/sample_cars.jpg --size m  # medium
```

## **🤖 Available Models**

### **YOLOv8n Object Detection**
- **Purpose**: Detect 80 different object classes
- **Input**: 640×640 RGB images
- **Output**: Bounding boxes with class labels and confidence scores
- **Use Cases**: General object detection, surveillance, autonomous vehicles

### **LPRNet License Plate Recognition**
- **Purpose**: Recognize license plate text
- **Input**: 24×94 RGB images (license plate regions)
- **Output**: Recognized text with character-level confidence
- **Use Cases**: Parking management, traffic enforcement, access control

### **Vehicle Detection and Tracking**
- **Purpose**: Detect and track vehicles across video frames
- **Input**: 416×416 RGB images
- **Output**: Vehicle bounding boxes with unique track IDs
- **Use Cases**: Traffic monitoring, parking analytics, fleet management

## **📜 License**
This project is licensed under the **MIT License**.

