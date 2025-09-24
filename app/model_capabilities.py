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
    # OpenVINO Object Detection Models
    "person-detection-retail-0013": {
        "type": "object_detection",
        "capabilities": {
            "detection_classes": ["person"],
            "output_format": "bounding_boxes",
            "confidence_threshold": 0.3,
            "description": "Person detection optimized for retail environments"
        }
    },
    "face-detection-retail-0005": {
        "type": "face_detection",
        "capabilities": {
            "detection_classes": ["face"],
            "output_format": "bounding_boxes",
            "confidence_threshold": 0.3,
            "description": "Face detection optimized for retail environments"
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
