import os

# Base directory for storing models
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Predefined models - all models must be committed to repository (no automatic download)
MODEL_URLS = {
    # YOLOv8 Object Detection (80 COCO classes)
    "yolov8n": {"cpu32": "local"},
    "yolov8s": {"cpu32": "local"},
    "yolov8m": {"cpu32": "local"},
    # OpenVINO Model Zoo - Face Detection
    "face-detection-retail-0005": {"cpu32": "local"},
    # OpenVINO Model Zoo - Vehicle & License Plate Detection
    "vehicle-license-plate-detection-barrier-0106": {"cpu32": "local"},
    # OpenVINO Model Zoo - Text Recognition (for license plates, signs)
    "text-recognition-0012": {"cpu32": "local"},
    # License Plate Recognition (Chinese plates - legacy)
    "lprnet": {"cpu32": "local"},
    # OpenVINO Model Zoo - Person Re-Identification (for tracking)
    "person-reidentification-retail-0286": {"cpu32": "local"}
}