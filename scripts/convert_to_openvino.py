#!/usr/bin/env python3
"""
Convert YOLOv8n PyTorch model to OpenVINO format.
Tries direct OpenVINO export first, then falls back to ONNX -> OpenVINO.
"""

import os
import openvino as ov
from ultralytics import YOLO


def convert_pytorch_to_openvino(model_path="yolov8n.pt", imgsz=640, with_nms=True):
    print("🚀 Starting YOLOv8n model conversion to OpenVINO...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    print(f"✅ Found {model_path} ({os.path.getsize(model_path)} bytes)")

    # Load YOLO model
    print("🔄 Loading YOLOv8 model...")
    model = YOLO(model_path)

    export_dir = os.path.dirname(model_path) or "."
    ov_model_path = os.path.join(export_dir, "yolov8n_openvino_model")

    try:
        # Try direct OpenVINO export (preferred)
        print("🔄 Exporting directly to OpenVINO format...")
        export_path = model.export(format="openvino", imgsz=imgsz, nms=with_nms)
        print(f"✅ Successfully exported to OpenVINO at {export_path}")
        return export_path

    except Exception as e:
        print(f"⚠️ Direct OpenVINO export failed: {e}")
        print("🔄 Falling back to ONNX -> OpenVINO...")

        try:
            # Export to ONNX first
            onnx_path = model.export(format="onnx", imgsz=imgsz, nms=with_nms, opset=16)
            if not onnx_path or not os.path.exists(onnx_path):
                raise RuntimeError("ONNX export did not produce a file")

            print(f"✅ Exported ONNX model: {onnx_path}")

            # Convert ONNX -> OpenVINO IR
            ov_model = ov.convert_model(onnx_path)
            xml_path = os.path.join(export_dir, "yolov8n.xml")
            ov.save_model(ov_model, xml_path)
            print(f"✅ Converted to OpenVINO IR: {xml_path}")
            return xml_path

        except Exception as e2:
            print(f"❌ Fallback conversion failed: {e2}")
            return None


if __name__ == "__main__":
    convert_pytorch_to_openvino()
