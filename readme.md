
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

## **📥 Model Setup & Management**

### **Automatic Model Download**
The easiest way to get all required models:

```sh
cd tests
python test_runner.py --download-models
```

### **Individual Model Downloads**
Download specific models as needed:

```sh
# YOLOv8 Object Detection Models
python test_runner.py --model yolov8n --download
python test_runner.py --model yolov8s --download  
python test_runner.py --model yolov8m --download

# Other Models
python test_runner.py --model lprnet --download
python test_runner.py --model vehicle_tracking --download
```

### **Manual Model Conversion**
If you have PyTorch models and need to convert them to OpenVINO:

```sh
# Convert YOLOv8 models
python scripts/convert_to_openvino.py --size n  # nano
python scripts/convert_to_openvino.py --size s  # small
python scripts/convert_to_openvino.py --size m  # medium

# Download and convert from scratch
python scripts/download_and_convert.py
```

### **Verify Model Installation**
Check if models are properly downloaded:

```sh
# Check YOLOv8 models
ls models/yolov8n/*.xml models/yolov8n/*.bin
ls models/yolov8s/*.xml models/yolov8s/*.bin
ls models/yolov8m/*.xml models/yolov8m/*.bin

# Check other models
ls models/lprnet/*.xml models/lprnet/*.bin
ls models/vehicle_tracking/*.xml models/vehicle_tracking/*.bin
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

