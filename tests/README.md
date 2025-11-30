# OpenVINO AI Testing Suite

Comprehensive testing suite for OpenVINO AI models including object detection, face detection, license plate detection & recognition, and person tracking.

## 📁 Directory Structure

```
tests/
├── ../models/                       # Model files (parent directory)
│   ├── yolov8n/                     # YOLOv8 object detection (80 COCO classes)
│   ├── face-detection-retail-0005/  # Face detection
│   ├── vehicle-license-plate-detection-barrier-0106/  # Vehicle & LP detection
│   ├── lprnet/                      # License plate text recognition
│   └── person-reidentification-retail-0286/  # Person re-ID for tracking
├── test_images/                     # Sample test images
├── test_videos/                     # Sample test videos
├── output/                          # Generated outputs
├── test_runner.py                   # Main test runner
├── test_yolov8_openvino.py          # YOLOv8 specific tests
├── test_vehicle_tracking.py         # Person/vehicle tracking
├── test_api.py                      # Docker API test suite
└── README.md                        # This file
```

## 🚀 Quick Start

```bash
# Setup virtual environment
cd services/ai-inference/tests
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# List available models
python test_runner.py --list-models

# Run tests
python test_runner.py --model yolov8n --input test_images/bus.jpg
```

---

## 📷 Image Detection Tests

### General Object Detection (YOLOv8n)

Detects 80 COCO object classes (person, car, dog, bicycle, etc.)

```bash
python test_runner.py --model yolov8n --input test_images/dog_bike_car.jpg

# Or use the dedicated script with more options
python test_yolov8_openvino.py --input test_images/dog_bike_car.jpg --output output/yolo_result.jpg
```

### Face Detection

Detects faces in images, optimized for retail environments.

```bash
# Using alias
python test_runner.py --model face --input test_images/bus.jpg

# Using full model name
python test_runner.py --model face-detection-retail-0005 --input test_images/bus.jpg
```

### Vehicle & License Plate Detection

Detects vehicles and license plates in barrier/gate scenarios.

```bash
# Using alias
python test_runner.py --model license_plate_detection --input test_images/licenseplate1.jpg
python test_runner.py --model lp_detection --input test_images/licenseplate1.jpg

# Using full model name
python test_runner.py --model vehicle-license-plate-detection-barrier-0106 --input test_images/licenseplate1.jpg
```

### License Plate Recognition (LPRNet)

Recognizes text from cropped license plate images.

```bash
python test_runner.py --model lprnet --input test_images/plate_crop.jpg
```

### Run All Image Tests

```bash
python test_runner.py --model all --input test_images/sample.jpg
```

---

## 🎬 Video Detection & Tracking Tests

### Object Detection on Video

```bash
python test_yolov8_openvino.py --input test_videos/traffic.mp4 --output output/traffic_detected.mp4

# With custom FPS
python test_yolov8_openvino.py --input test_videos/traffic.mp4 --inference-fps 5
```

### Vehicle/Person Tracking

Uses YOLOv8n for detection + tracking algorithms.

```bash
# IoU-based tracking (simple, fast)
python test_vehicle_tracking.py --input test_videos/pedestrians.mp4 --tracker iou

# ByteTrack tracking (handles occlusions better)
python test_vehicle_tracking.py --input test_videos/pedestrians.mp4 --tracker bytetrack

# Custom output path
python test_vehicle_tracking.py --input video.mp4 --tracker bytetrack --output tracked_output.mp4
```

---

## 📊 Available Models

| Model | Alias | Type | Description |
|-------|-------|------|-------------|
| `yolov8n` | - | Object Detection | 80 COCO classes |
| `face-detection-retail-0005` | `face` | Face Detection | Retail optimized |
| `vehicle-license-plate-detection-barrier-0106` | `license_plate_detection`, `lp_detection` | LP Detection | Vehicle + plate |
| `lprnet` | - | OCR | License plate text |
| `person-reidentification-retail-0286` | `person_reid`, `reid` | Re-ID | Person appearance matching |

### Model Aliases

For convenience, you can use short aliases instead of full model names:

```bash
# These are equivalent:
python test_runner.py --model face --input image.jpg
python test_runner.py --model face-detection-retail-0005 --input image.jpg

# These are equivalent:
python test_runner.py --model lp_detection --input image.jpg
python test_runner.py --model vehicle-license-plate-detection-barrier-0106 --input image.jpg
```

---

## 🐳 Docker API Testing

Test the FastAPI-based AI inference service running in Docker.

### Setup

```bash
# Build and run the Docker container
cd services/ai-inference
docker build -t ai-inference .
docker run -d -p 8001:8001 --name ai-inference ai-inference

# Verify it's running
curl http://localhost:8001/health
```

### Run API Tests

```bash
cd tests

# Run all API tests
python test_api.py --all

# Test against specific host/port
python test_api.py --host localhost --port 8001

# Test with specific image
python test_api.py --image test_images/dog_bike_car.jpg

# Run performance benchmark
python test_api.py --benchmark --iterations 10
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/inference/detection` | POST | Object detection |
| `/inference/direct` | POST | Raw model inference |

### Example API Calls

```bash
# Health check
curl http://localhost:8001/health

# List models
curl http://localhost:8001/models

# Object detection
curl -X POST "http://localhost:8001/inference/detection?object_name=car" \
  -F "image=@test_images/sample.jpg"

# Face detection
curl -X POST "http://localhost:8001/inference/detection?object_name=face" \
  -F "image=@test_images/people.jpg"
```

---

## 🛠 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Check `../models/<model_name>/` has `.xml` and `.bin` files |
| Wrong class labels | Verify `model.json` has correct `classes` array |
| Too many detections | Increase `confidence_threshold` in model config |
| Tracking ID jumps | Lower `iou_threshold` or use ByteTrack |
| Docker connection refused | Run `docker ps` to verify container is running |
| Input shape mismatch | Model auto-detects NCHW vs NHWC format |

---

## 📄 License

Models from OpenVINO Model Zoo are under Apache-2.0 license.
