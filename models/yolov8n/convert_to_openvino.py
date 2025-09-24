#!/usr/bin/env python3
"""
Script to convert YOLOv8n PyTorch model to OpenVINO format using OpenVINO tools
"""

import os
import sys
from pathlib import Path

def convert_pytorch_to_openvino():
    """Convert PyTorch model to OpenVINO format using OpenVINO tools"""
    try:
        # Try to import OpenVINO
        import openvino as ov
        
        # Try to load the PyTorch model
        print("🔄 Loading PyTorch model...")
        
        # For now, let's try a different approach - create a simple ONNX model first
        # and then convert to OpenVINO
        
        # Try to use ultralytics to export to ONNX first
        try:
            from ultralytics import YOLO
            
            # Load the model
            model = YOLO('yolov8n.pt')
            
            # Export to ONNX
            print("🔄 Exporting to ONNX format...")
            model.export(format='onnx', imgsz=640)
            
            if os.path.exists('yolov8n.onnx'):
                print("✅ Successfully exported to ONNX")
                
                # Now convert ONNX to OpenVINO
                print("🔄 Converting ONNX to OpenVINO...")
                ov_model = ov.convert_model('yolov8n.onnx')
                
                # Save the OpenVINO model
                ov.save_model(ov_model, 'yolov8n.xml')
                print("✅ Successfully converted to OpenVINO format")
                
                return True
            else:
                print("❌ ONNX export failed")
                return False
                
        except ImportError:
            print("❌ ultralytics not available")
            return False
        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            return False
            
    except ImportError:
        print("❌ OpenVINO not available")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Starting YOLOv8n model conversion to OpenVINO...")
    
    # Change to the model directory
    os.chdir(Path(__file__).parent)
    
    # Check if PyTorch model exists
    if not os.path.exists('yolov8n.pt'):
        print("❌ YOLOv8n.pt not found")
        return False
    
    print(f"✅ Found YOLOv8n.pt ({os.path.getsize('yolov8n.pt')} bytes)")
    
    # Convert to OpenVINO format
    if not convert_pytorch_to_openvino():
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
