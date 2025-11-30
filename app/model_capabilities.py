"""
Model Capabilities Registry for OpenVINO
Defines the capabilities and output formats for different model types.
"""

# COCO classes for object detection models
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", 
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Model capabilities registry for OpenVINO models
MODEL_CAPABILITIES = {
    # YOLOv8 Object Detection Models (converted to OpenVINO)
    "yolov8n": {
        "type": "object_detection",
        "capabilities": {
            "detection_classes": COCO_CLASSES,
            "num_classes": 80,
            "output_format": "bounding_boxes",
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "input_size": [640, 640],
            "description": "YOLOv8 Nano - Fast object detection (80 COCO classes)"
        }
    },
    "yolov8s": {
        "type": "object_detection",
        "capabilities": {
            "detection_classes": COCO_CLASSES,
            "num_classes": 80,
            "output_format": "bounding_boxes",
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "input_size": [640, 640],
            "description": "YOLOv8 Small - Balanced speed and accuracy (80 COCO classes)"
        }
    },
    "yolov8m": {
        "type": "object_detection",
        "capabilities": {
            "detection_classes": COCO_CLASSES,
            "num_classes": 80,
            "output_format": "bounding_boxes",
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "input_size": [640, 640],
            "description": "YOLOv8 Medium - High accuracy (80 COCO classes)"
        }
    },
    # OpenVINO Model Zoo - Face Detection
    "face-detection-retail-0005": {
        "type": "face_detection",
        "capabilities": {
            "detection_classes": ["face"],
            "num_classes": 1,
            "output_format": "bounding_boxes",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "input_size": [300, 300],
            "description": "Face detection optimized for retail environments (OpenVINO Model Zoo)"
        }
    },
    # OpenVINO Model Zoo - Vehicle & License Plate Detection
    "vehicle-license-plate-detection-barrier-0106": {
        "type": "license_plate_detection",
        "capabilities": {
            "detection_classes": ["vehicle", "license_plate"],
            "num_classes": 2,
            "output_format": "bounding_boxes",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "input_size": [300, 300],
            "description": "Vehicle and license plate detection for barrier/gate scenarios (OpenVINO Model Zoo)"
        }
    },
    # Text Recognition (for license plates, signs, etc.) - Western alphanumeric
    "text-recognition-0012": {
        "type": "text_recognition",
        "capabilities": {
            "output_format": "text",
            "input_size": [120, 32],
            "character_set": "0-9, A-Z",
            "description": "General text recognition for alphanumeric sequences (Intel OpenVINO)"
        }
    },
    # License Plate Recognition (Chinese plates only - legacy)
    "lprnet": {
        "type": "license_plate_recognition",
        "capabilities": {
            "output_format": "text",
            "input_size": [94, 24],
            "description": "Chinese License Plate Recognition (not for Western plates)"
        }
    },
    # Person Re-Identification (for tracking across frames/cameras)
    "person-reidentification-retail-0286": {
        "type": "person_reidentification",
        "capabilities": {
            "output_format": "embedding",
            "embedding_size": 256,
            "input_size": [128, 256],
            "description": "Person re-identification for tracking - generates appearance embeddings"
        }
    }
}

def get_model_capabilities(model_name: str):
    """Get capabilities for a specific model."""
    return MODEL_CAPABILITIES.get(model_name, {
        "type": "unknown",
        "capabilities": {
            "description": f"Unknown model: {model_name}"
        }
    })

def get_models_by_type(model_type: str):
    """Get all models of a specific type."""
    return [
        model_name for model_name, info in MODEL_CAPABILITIES.items()
        if info["type"] == model_type
    ]

def get_detailed_model_info():
    """Get detailed information about all models."""
    detailed_info = {}
    for model_name, capabilities in MODEL_CAPABILITIES.items():
        detailed_info[model_name] = {
            "name": model_name,
            "type": capabilities["type"],
            "capabilities": capabilities["capabilities"],
            "input_shape": [640, 640],  # Default shape for OpenVINO models
            "architecture": "openvino"
        }
    
    return detailed_info

def get_available_model_types():
    """Get all available model types."""
    return list(set(info["type"] for info in MODEL_CAPABILITIES.values()))

def get_available_objects():
    """Get all available object types for detection (legacy compatibility)."""
    objects = set()
    for model_name, capabilities in MODEL_CAPABILITIES.items():
        if "detection_classes" in capabilities["capabilities"]:
            objects.update(capabilities["capabilities"]["detection_classes"])
    return sorted(list(objects))
