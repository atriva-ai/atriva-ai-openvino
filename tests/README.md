# OpenVINO AI Testing Suite

This directory contains a comprehensive testing suite for OpenVINO AI models including YOLOv8n object detection, LPRNet license plate recognition, and vehicle detection and tracking.

## 📁 Directory Structure

```
tests/
├── ../models/                       # Model configurations and downloaded files (top-level)
│   ├── yolov8n/
│   │   ├── model.json              # YOLOv8n configuration
│   │   ├── yolov8n.xml             # OpenVINO XML model file
│   │   └── yolov8n.bin             # OpenVINO BIN model file
│   ├── lprnet/
│   │   ├── model.json              # LPRNet configuration
│   │   ├── lprnet.xml              # OpenVINO XML model file
│   │   └── lprnet.bin              # OpenVINO BIN model file
│   └── vehicle_tracking/
│       ├── model.json              # Vehicle tracking configuration
│       ├── vehicle_tracking.xml    # OpenVINO XML model file
│       └── vehicle_tracking.bin    # OpenVINO BIN model file
├── test_images/                     # Sample test images
├── test_videos/                     # Sample test videos
├── output/                          # Generated annotated outputs
├── venv/                           # Python virtual environment
├── requirements.txt                 # Python dependencies
├── setup.sh                        # Environment setup script
├── test_runner.py                  # Main test runner
├── test_yolov8n.py                 # YOLOv8n specific tests
├── test_lprnet.py                  # LPRNet specific tests
├── test_vehicle_tracking.py        # Vehicle tracking specific tests
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run the setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Download Models

```bash
# Download all models
python test_runner.py --download-models

# Or download individual models
python test_runner.py --model yolov8n --download
```

### 3. Run Tests

```bash
# Test all models on an image
python test_runner.py --model all --input test_images/sample_cars.jpg

# Test specific model
python test_runner.py --model yolov8n --input test_images/sample_cars.jpg

# Test on video
python test_runner.py --model all --input test_videos/sample_traffic.mp4
```

## 🧪 Individual Model Tests

### YOLOv8n Object Detection

```bash
# Test on image
python test_yolov8n.py --input test_images/sample_cars.jpg

# Test on video
python test_yolov8n.py --input test_videos/sample_traffic.mp4 --video

# Custom output path
python test_yolov8n.py --input test_images/sample_cars.jpg --output my_output.jpg
```

**Features:**
- Detects 80 different object classes
- Processes images and videos
- Generates annotated outputs with bounding boxes
- Configurable confidence thresholds

### LPRNet License Plate Recognition

```bash
# Test on image
python test_lprnet.py --input test_images/sample_license_plate.jpg

# Test on video
python test_lprnet.py --input test_videos/sample_parking.mp4 --video
```

**Features:**
- Recognizes license plate text
- Handles multiple license plates per image
- Processes video streams
- Character-level confidence scoring

### Vehicle Detection and Tracking

```bash
# Test on image
python test_vehicle_tracking.py --input test_images/sample_vehicles.jpg

# Test on video
python test_vehicle_tracking.py --input test_videos/sample_traffic.mp4 --video
```

**Features:**
- Detects vehicles (cars, trucks, buses, motorcycles, bicycles)
- Tracks vehicles across video frames
- Assigns unique track IDs
- Color-coded tracking visualization

## 📊 Model Configurations

### YOLOv8n Configuration

```json
{
  "model_name": "yolov8n",
  "model_type": "object_detection",
  "input_shape": [1, 3, 640, 640],
  "classes": ["person", "bicycle", "car", "motorcycle", ...],
  "confidence_threshold": 0.25,
  "nms_threshold": 0.45
}
```

### LPRNet Configuration

```json
{
  "model_name": "lprnet",
  "model_type": "license_plate_recognition",
  "input_shape": [1, 3, 24, 94],
  "classes": ["0", "1", "2", ..., "A", "B", "C", ...],
  "confidence_threshold": 0.5,
  "max_plate_length": 18
}
```

### Vehicle Tracking Configuration

```json
{
  "model_name": "vehicle_tracking",
  "model_type": "vehicle_detection_and_tracking",
  "input_shape": [1, 3, 416, 416],
  "classes": ["car", "truck", "bus", "motorcycle", "bicycle", "person"],
  "confidence_threshold": 0.3,
  "tracking_threshold": 0.5,
  "max_tracks": 50
}
```

## 🔧 Advanced Usage

### Custom Model Testing

```python
from test_runner import OpenVINOModelTester

# Initialize tester
tester = OpenVINOModelTester()

# Load custom model
model, config = tester.load_openvino_model("custom_model")

# Run custom inference
results = tester.run_custom_inference(input_data)
```

### Batch Processing

```bash
# Process multiple images
for image in test_images/*.jpg; do
    python test_runner.py --model yolov8n --input "$image"
done
```

### Performance Benchmarking

```bash
# Test inference speed
python test_runner.py --model all --input test_images/sample_cars.jpg --benchmark
```

## 📈 Output Examples

### YOLOv8n Output
- **Input**: `test_images/sample_cars.jpg`
- **Output**: `output/yolov8n_detection_cars.jpg`
- **Features**: Bounding boxes with class labels and confidence scores

### LPRNet Output
- **Input**: `test_images/sample_license_plate.jpg`
- **Output**: `output/lprnet_recognition_plate.jpg`
- **Features**: License plate regions with recognized text

### Vehicle Tracking Output
- **Input**: `test_videos/sample_traffic.mp4`
- **Output**: `output/vehicle_tracking_traffic.mp4`
- **Features**: Tracked vehicles with unique IDs and trajectories

## 🛠 Troubleshooting

### Common Issues

1. **Model files not found**
   ```bash
   # Download missing models
   python test_runner.py --download-models
   ```

2. **OpenVINO installation issues**
   ```bash
   # Reinstall OpenVINO
   pip uninstall openvino
   pip install openvino>=2023.0.0
   ```

3. **Memory issues**
   ```bash
   # Reduce batch size or image resolution
   export OPENVINO_THREADS=4
   ```

### Performance Optimization

1. **CPU Optimization**
   ```bash
   # Use CPU threading
   export OPENVINO_THREADS=8
   ```

2. **Memory Management**
   ```bash
   # Limit memory usage
   export OPENVINO_MEMORY_POOL_SIZE=1024
   ```

## 📝 Adding New Models

1. **Create model directory**
   ```bash
   mkdir models/new_model
   ```

2. **Add model configuration**
   ```json
   {
     "model_name": "new_model",
     "model_type": "custom_type",
     "input_shape": [1, 3, 224, 224],
     "classes": ["class1", "class2"],
     "confidence_threshold": 0.5
   }
   ```

3. **Download model files**
   ```bash
   # Add download URLs to model.json
   # Run: python test_runner.py --download-models
   ```

4. **Create test script**
   ```python
   # Copy and modify existing test script
   cp test_yolov8n.py test_new_model.py
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your model tests
4. Update documentation
5. Submit a pull request

## 📄 License

This testing suite is part of the Atriva AI project and follows the same licensing terms.

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the OpenVINO documentation
- Contact the development team
