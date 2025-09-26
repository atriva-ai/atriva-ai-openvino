#!/usr/bin/env python3
"""
Create model.json configuration file for YOLOv8 models.
This script can be used to generate model.json for existing PyTorch or OpenVINO models.
"""

import os
import json
import argparse
from pathlib import Path
from ultralytics import YOLO


def create_model_json(model_path, model_size='n', output_dir=None, custom_classes=None):
    """Create model.json configuration file"""
    if output_dir is None:
        output_dir = os.path.dirname(model_path) or "."
    
    # Load the model to get information
    try:
        model = YOLO(model_path)
        
        # Get model info
        model_info = model.info()
        
        # Extract classes from the model itself
        if custom_classes:
            classes = custom_classes
        else:
            # Try to get classes from the model
            try:
                # For fine-tuned models, get classes from model.names
                if hasattr(model.model, 'names') and model.model.names:
                    classes = list(model.model.names.values())
                    print(f"✅ Extracted {len(classes)} classes from model")
                else:
                    # Fallback: try to get from model metadata
                    if hasattr(model, 'names') and model.names:
                        classes = list(model.names.values())
                        print(f"✅ Extracted {len(classes)} classes from model metadata")
                    else:
                        # Last resort: use default COCO classes
                        print("⚠️ Could not extract classes from model, using default COCO classes")
                        classes = [
                            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                        ]
            except Exception as e:
                print(f"⚠️ Error extracting classes from model: {e}")
                print("Using default COCO classes")
                classes = [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                ]
        
        # Create model configuration
        model_config = {
            "model_name": f"yolov8{model_size}",
            "model_type": "object_detection",
            "architecture": "openvino",
            "input_shape": [1, 3, 640, 640],
            "output_shape": [1, 84, 8400],  # Standard YOLOv8 output shape
            "precision": "FP32",
            "classes": classes,
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "description": f"YOLOv8 {model_size.upper()} - Fine-tuned object detection model optimized for OpenVINO",
            "download_url": f"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8{model_size}.onnx",
            "openvino_model_url": f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/yolov8{model_size}/FP32/",
            "license": "GPL-3.0",
            "author": "Ultralytics",
            "version": "8.0.0",
            "fine_tuned": True,
            "source_model": os.path.basename(model_path)
        }
        
        # Save model.json
        json_path = os.path.join(output_dir, "model.json")
        with open(json_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✅ Created model.json: {json_path}")
        print(f"📊 Model info:")
        print(f"   - Model name: {model_config['model_name']}")
        print(f"   - Classes: {len(classes)}")
        print(f"   - Input shape: {model_config['input_shape']}")
        print(f"   - Output shape: {model_config['output_shape']}")
        print(f"   - Confidence threshold: {model_config['confidence_threshold']}")
        print(f"   - NMS threshold: {model_config['nms_threshold']}")
        
        # Print the actual classes for verification
        print(f"📋 Classes found in model:")
        for i, class_name in enumerate(classes):
            print(f"   {i}: {class_name}")
        
        return json_path
        
    except Exception as e:
        print(f"❌ Error creating model.json: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create model.json configuration file for YOLOv8 models")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to PyTorch model file (.pt)")
    parser.add_argument("--size", choices=['n', 's', 'm'], default='n', 
                       help="YOLOv8 model size: n (nano), s (small), m (medium)")
    parser.add_argument("--output-dir", type=str, 
                       help="Output directory for model.json (default: same as model)")
    parser.add_argument("--classes", type=str, 
                       help="Path to JSON file with custom class names")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Confidence threshold (default: 0.25)")
    parser.add_argument("--nms", type=float, default=0.45,
                       help="NMS threshold (default: 0.45)")
    
    args = parser.parse_args()
    
    # Load custom classes if provided
    custom_classes = None
    if args.classes:
        try:
            with open(args.classes, 'r') as f:
                custom_classes = json.load(f)
            print(f"✅ Loaded custom classes from: {args.classes}")
        except Exception as e:
            print(f"❌ Error loading custom classes: {e}")
            return False
    
    # Create model.json
    json_path = create_model_json(
        args.model, 
        args.size, 
        args.output_dir,
        custom_classes
    )
    
    if json_path:
        # Update thresholds if provided
        if args.confidence != 0.25 or args.nms != 0.45:
            try:
                with open(json_path, 'r') as f:
                    config = json.load(f)
                config['confidence_threshold'] = args.confidence
                config['nms_threshold'] = args.nms
                with open(json_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Updated thresholds: confidence={args.confidence}, nms={args.nms}")
            except Exception as e:
                print(f"⚠️ Error updating thresholds: {e}")
        
        print(f"🎉 Successfully created model.json!")
        return True
    else:
        print("❌ Failed to create model.json")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
