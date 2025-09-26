#!/usr/bin/env python3
"""
Script to download YOLOv8n PyTorch model and convert it to OpenVINO format
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def download_yolov8n_pt():
    """Download YOLOv8n PyTorch model using ultralytics"""
    try:
        # Try to import ultralytics
        from ultralytics import YOLO
        
        # Download the model
        model = YOLO('yolov8n.pt')
        print("✅ Successfully downloaded YOLOv8n.pt")
        return True
        
    except ImportError:
        print("❌ ultralytics not available, trying alternative download...")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def download_alternative():
    """Alternative download method"""
    try:
        # Try different URLs
        urls = [
            "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
        ]
        
        for url in urls:
            try:
                print(f"Trying to download from: {url}")
                response = requests.get(url, timeout=30)
                if response.status_code == 200 and len(response.content) > 1000:  # Check if it's not an error page
                    with open('yolov8n.pt', 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Successfully downloaded YOLOv8n.pt from {url}")
                    return True
            except Exception as e:
                print(f"❌ Failed to download from {url}: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"❌ Error in alternative download: {e}")
        return False

def convert_to_openvino():
    """Convert PyTorch model to OpenVINO format"""
    try:
        # Try to import ultralytics
        from ultralytics import YOLO
        
        # Load the model
        model = YOLO('yolov8n.pt')
        
        # Export to OpenVINO format
        model.export(format='openvino', imgsz=640)
        
        print("✅ Successfully converted to OpenVINO format")
        return True
        
    except ImportError:
        print("❌ ultralytics not available for conversion")
        return False
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Starting YOLOv8n model download and conversion...")
    
    # Change to the model directory
    os.chdir(Path(__file__).parent)
    
    # Try to download the model
    if not download_yolov8n_pt():
        if not download_alternative():
            print("❌ Failed to download YOLOv8n.pt")
            return False
    
    # Check if the file exists and is valid
    if not os.path.exists('yolov8n.pt') or os.path.getsize('yolov8n.pt') < 1000:
        print("❌ Downloaded file is invalid or too small")
        return False
    
    # Convert to OpenVINO format
    if not convert_to_openvino():
        print("❌ Failed to convert to OpenVINO format")
        return False
    
    # Check if conversion was successful
    if os.path.exists('yolov8n.xml') and os.path.exists('yolov8n.bin'):
        print("✅ Successfully created YOLOv8n OpenVINO model files")
        print(f"   - yolov8n.xml: {os.path.getsize('yolov8n.xml')} bytes")
        print(f"   - yolov8n.bin: {os.path.getsize('yolov8n.bin')} bytes")
        return True
    else:
        print("❌ OpenVINO model files not found after conversion")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
