#!/usr/bin/env python3
"""
OpenVINO AI Model Testing Suite
===============================

This script provides comprehensive testing for OpenVINO AI models including:
- YOLOv8n Object Detection
- LPRNet License Plate Recognition  
- Vehicle Detection and Tracking

Usage:
    python test_runner.py --model yolov8n --input test_images/sample.jpg
    python test_runner.py --model all --input test_videos/sample.mp4
    python test_runner.py --download-models
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from openvino.runtime import Core
import requests
import yaml
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class OpenVINOModelTester:
    """Main testing class for OpenVINO AI models"""
    
    def __init__(self, models_dir: str = "../models"):
        self.models_dir = Path(models_dir)
        self.test_images_dir = Path("test_images")
        self.test_videos_dir = Path("test_videos")
        self.output_dir = Path("output")
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Initialize OpenVINO Core
        self.ie = Core()
        self.available_devices = self.ie.available_devices
        console.print(f"[green]Available OpenVINO devices: {self.available_devices}[/green]")
        
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.models_dir,
            self.test_images_dir,
            self.test_videos_dir,
            self.output_dir
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            console.print(f"[blue]Created directory: {directory}[/blue]")
    
    def load_model_config(self, model_name: str) -> Dict:
        """Load model configuration from JSON file"""
        config_path = self.models_dir / model_name / "model.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def download_model(self, model_name: str) -> bool:
        """Download and convert model to OpenVINO format"""
        config = self.load_model_config(model_name)
        model_dir = self.models_dir / model_name
        
        console.print(f"[yellow]Downloading {model_name} model...[/yellow]")
        
        try:
            # For YOLOv8 models, use ultralytics to download and convert
            if model_name.startswith("yolov8"):
                return self._download_and_convert_yolov8(model_name, model_dir)
            
            # For other models, try direct download from OpenVINO model zoo
            base_url = config.get("openvino_model_url", "").rstrip("/")
            if not base_url:
                console.print(f"[red]❌ No download URL configured for {model_name}[/red]")
                return False
                
            xml_url = f"{base_url}/{model_name}.xml"
            xml_path = model_dir / f"{model_name}.xml"
            
            response = requests.get(xml_url, stream=True)
            response.raise_for_status()
            
            with open(xml_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading XML"):
                    f.write(chunk)
            
            # Download BIN file
            bin_url = f"{base_url}/{model_name}.bin"
            bin_path = model_dir / f"{model_name}.bin"
            
            response = requests.get(bin_url, stream=True)
            response.raise_for_status()
            
            with open(bin_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading BIN"):
                    f.write(chunk)
            
            console.print(f"[green]✅ Successfully downloaded {model_name} model[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to download {model_name}: {str(e)}[/red]")
            return False
    
    def _download_and_convert_yolov8(self, model_name: str, model_dir: Path) -> bool:
        """Download YOLOv8 model and convert to OpenVINO format using ultralytics"""
        try:
            from ultralytics import YOLO
        except ImportError:
            console.print("[red]❌ ultralytics package not installed. Run: pip install ultralytics[/red]")
            return False
        
        try:
            console.print(f"[yellow]Downloading {model_name} from Ultralytics...[/yellow]")
            
            # Load the model (this downloads the .pt file automatically)
            model = YOLO(f"{model_name}.pt")
            
            # Export to OpenVINO format
            console.print(f"[yellow]Converting {model_name} to OpenVINO format...[/yellow]")
            export_path = model.export(format="openvino")
            
            # Move the exported files to our model directory
            export_dir = Path(export_path)
            xml_src = export_dir / f"{model_name}.xml"
            bin_src = export_dir / f"{model_name}.bin"
            
            xml_dst = model_dir / f"{model_name}.xml"
            bin_dst = model_dir / f"{model_name}.bin"
            
            # Copy files to model directory
            import shutil
            if xml_src.exists():
                shutil.copy2(xml_src, xml_dst)
            if bin_src.exists():
                shutil.copy2(bin_src, bin_dst)
            
            console.print(f"[green]✅ Successfully downloaded and converted {model_name} to OpenVINO format[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to download/convert {model_name}: {str(e)}[/red]")
            return False
    
    def _load_yolov8_classes(self, model_name: str) -> List[str]:
        """Load class names from ultralytics metadata.yaml file"""
        model_dir = self.models_dir / model_name
        
        # Search for metadata.yaml - prioritize ultralytics subdirectory (COCO model)
        search_paths = [
            model_dir / "ultralytics" / "metadata.yaml",
            model_dir / f"{model_name}_openvino_model" / "metadata.yaml",
            model_dir / "metadata.yaml",  # Fallback to root (may be custom-trained)
        ]
        
        for metadata_path in search_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    
                    # Extract class names from 'names' dict
                    names_dict = metadata.get('names', {})
                    if names_dict:
                        # Convert dict {0: 'person', 1: 'bicycle', ...} to list
                        max_idx = max(names_dict.keys())
                        classes = [''] * (max_idx + 1)
                        for idx, name in names_dict.items():
                            classes[idx] = name
                        console.print(f"[blue]Loaded {len(classes)} class names from {metadata_path}[/blue]")
                        return classes
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load metadata from {metadata_path}: {e}[/yellow]")
        
        return []
    
    def load_openvino_model(self, model_name: str):
        """Load OpenVINO model for inference"""
        config = self.load_model_config(model_name)
        model_dir = self.models_dir / model_name
        
        xml_path = model_dir / f"{model_name}.xml"
        bin_path = model_dir / f"{model_name}.bin"
        
        if not xml_path.exists() or not bin_path.exists():
            console.print(f"[red]Model files not found for {model_name}. Downloading...[/red]")
            if not self.download_model(model_name):
                raise FileNotFoundError(f"Could not download {model_name} model")
        
        # Load model
        model = self.ie.read_model(model=xml_path)
        compiled_model = self.ie.compile_model(model=model, device_name="CPU")
        
        console.print(f"[green]✅ Loaded {model_name} model successfully[/green]")
        return compiled_model, config
    
    def test_yolov8n_detection(self, input_path: str, output_path: str) -> bool:
        """Test YOLOv8n object detection"""
        console.print("[blue]Testing YOLOv8n Object Detection...[/blue]")
        
        try:
            model, config = self.load_openvino_model("yolov8n")
            
            # Load and preprocess image
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            orig_h, orig_w = image.shape[:2]
            
            # Resize image to model input size
            input_shape = config["input_shape"][2:]  # [640, 640]
            input_h, input_w = input_shape
            resized_image = cv2.resize(image, (input_w, input_h))
            
            # Convert BGR to RGB and normalize
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            normalized_image = rgb_image.astype(np.float32) / 255.0
            
            # Transpose to CHW format
            input_data = np.transpose(normalized_image, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            output_tensor = model.output(0)
            
            result = model(input_data)
            outputs = result[output_tensor]
            
            console.print(f"[green]✅ YOLOv8n inference completed. Output shape: {outputs.shape}[/green]")
            
            # Post-process YOLOv8 outputs
            # Output shape is (1, 84, 8400) -> transpose to (8400, 84)
            predictions = outputs[0].T  # Shape: (8400, 84)
            
            conf_threshold = config.get("confidence_threshold", 0.25)
            nms_threshold = config.get("nms_threshold", 0.45)
            
            # Load class names from metadata.yaml (ultralytics export) if available
            classes = self._load_yolov8_classes("yolov8n")
            if not classes:
                classes = config.get("classes", [])
            
            boxes = []
            confidences = []
            class_ids = []
            
            # Scale factors
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            
            for pred in predictions:
                # First 4 values: cx, cy, w, h
                cx, cy, w, h = pred[:4]
                # Remaining 80 values: class scores
                class_scores = pred[4:]
                
                # Get best class
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                
                if confidence > conf_threshold:
                    # Convert from center format to corner format and scale to original image
                    x1 = int((cx - w / 2) * scale_x)
                    y1 = int((cy - h / 2) * scale_y)
                    box_w = int(w * scale_x)
                    box_h = int(h * scale_y)
                    
                    boxes.append([x1, y1, box_w, box_h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
            
            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            
            # Draw bounding boxes
            annotated_image = image.copy()
            
            # Generate colors for each class
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)
            
            detection_count = 0
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    
                    # Get class name and color
                    class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                    color = tuple(map(int, colors[class_id % len(colors)]))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label background
                    label = f"{class_name}: {confidence:.2f}"
                    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_image, (x, y - label_h - 10), (x + label_w, y), color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated_image, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    detection_count += 1
            
            console.print(f"[green]✅ Detected {detection_count} objects[/green]")
            
            cv2.imwrite(output_path, annotated_image)
            console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ YOLOv8n test failed: {str(e)}[/red]")
            return False
    
    def test_lprnet_recognition(self, input_path: str, output_path: str) -> bool:
        """Test LPRNet license plate recognition"""
        console.print("[blue]Testing LPRNet License Plate Recognition...[/blue]")
        
        try:
            model, config = self.load_openvino_model("lprnet")
            
            # Load and preprocess image
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            # Resize image to model input size
            input_shape = config["input_shape"][2:]  # [24, 94]
            resized_image = cv2.resize(image, input_shape)
            
            # Convert BGR to RGB and normalize
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            normalized_image = rgb_image.astype(np.float32) / 255.0
            
            # Transpose to CHW format
            input_data = np.transpose(normalized_image, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            output_tensor = model.output(0)
            
            result = model(input_data)
            outputs = result[output_tensor]
            
            # Process outputs (simplified - would need proper post-processing)
            console.print(f"[green]✅ LPRNet inference completed. Output shape: {outputs.shape}[/green]")
            
            # Save annotated image
            annotated_image = image.copy()
            cv2.putText(annotated_image, "LPRNet Recognition", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite(output_path, annotated_image)
            console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ LPRNet test failed: {str(e)}[/red]")
            return False
    
    def test_vehicle_tracking(self, input_path: str, output_path: str) -> bool:
        """Test vehicle detection and tracking"""
        console.print("[blue]Testing Vehicle Detection and Tracking...[/blue]")
        
        try:
            model, config = self.load_openvino_model("vehicle_tracking")
            
            # Load and preprocess image
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            # Resize image to model input size
            input_shape = config["input_shape"][2:]  # [416, 416]
            resized_image = cv2.resize(image, input_shape)
            
            # Convert BGR to RGB and normalize
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            normalized_image = rgb_image.astype(np.float32) / 255.0
            
            # Transpose to CHW format
            input_data = np.transpose(normalized_image, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            output_tensor = model.output(0)
            
            result = model(input_data)
            outputs = result[output_tensor]
            
            # Process outputs (simplified - would need proper post-processing)
            console.print(f"[green]✅ Vehicle tracking inference completed. Output shape: {outputs.shape}[/green]")
            
            # Save annotated image
            annotated_image = image.copy()
            cv2.putText(annotated_image, "Vehicle Tracking", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite(output_path, annotated_image)
            console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Vehicle tracking test failed: {str(e)}[/red]")
            return False
    
    def test_video_processing(self, model_name: str, input_path: str, output_path: str) -> bool:
        """Test video processing with specified model"""
        console.print(f"[blue]Testing {model_name} on video: {input_path}[/blue]")
        
        try:
            # Only YOLOv8n is fully implemented for video
            if model_name != "yolov8n":
                console.print(f"[yellow]⚠️ Video processing only implemented for yolov8n, skipping {model_name}[/yellow]")
                return False
            
            # Load model and config
            model, config = self.load_openvino_model(model_name)
            classes = self._load_yolov8_classes(model_name)
            if not classes:
                classes = config.get("classes", [])
            
            conf_threshold = config.get("confidence_threshold", 0.25)
            nms_threshold = config.get("nms_threshold", 0.45)
            input_shape = config["input_shape"][2:]  # [640, 640]
            input_h, input_w = input_shape
            
            # Generate colors for each class
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            console.print(f"[blue]Video: {width}x{height} @ {fps}fps, {total_frames} frames[/blue]")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            output_tensor = model.output(0)
            scale_x = width / input_w
            scale_y = height / input_h
            
            frame_count = 0
            for _ in tqdm(range(total_frames), desc="Processing frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                resized = cv2.resize(frame, (input_w, input_h))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                normalized = rgb.astype(np.float32) / 255.0
                input_data = np.transpose(normalized, (2, 0, 1))
                input_data = np.expand_dims(input_data, axis=0)
                
                # Run inference
                result = model(input_data)
                outputs = result[output_tensor]
                predictions = outputs[0].T
                
                # Post-process detections
                boxes, confidences, class_ids = [], [], []
                for pred in predictions:
                    cx, cy, w, h = pred[:4]
                    class_scores = pred[4:]
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id]
                    
                    if confidence > conf_threshold:
                        x1 = int((cx - w / 2) * scale_x)
                        y1 = int((cy - h / 2) * scale_y)
                        box_w = int(w * scale_x)
                        box_h = int(h * scale_y)
                        boxes.append([x1, y1, box_w, box_h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                
                # Apply NMS and draw boxes
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        class_id = class_ids[i]
                        confidence = confidences[i]
                        class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                        color = tuple(map(int, colors[class_id % len(colors)]))
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x, y - lh - 10), (x + lw, y), color, -1)
                        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            console.print(f"[green]✅ Processed video: {frame_count} frames saved to {output_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Video processing failed: {str(e)}[/red]")
            return False
    
    def run_all_tests(self, input_path: str) -> Dict[str, bool]:
        """Run all model tests"""
        results = {}
        
        # Determine if input is image or video
        is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        if is_video:
            # Test video processing
            for model_name in ["yolov8n", "lprnet", "vehicle_tracking"]:
                output_path = self.output_dir / f"{model_name}_video_output.mp4"
                results[model_name] = self.test_video_processing(model_name, input_path, str(output_path))
        else:
            # Test image processing
            results["yolov8n"] = self.test_yolov8n_detection(input_path, 
                str(self.output_dir / "yolov8n_detection.jpg"))
            results["lprnet"] = self.test_lprnet_recognition(input_path, 
                str(self.output_dir / "lprnet_recognition.jpg"))
            results["vehicle_tracking"] = self.test_vehicle_tracking(input_path, 
                str(self.output_dir / "vehicle_tracking.jpg"))
        
        return results
    
    def download_all_models(self) -> bool:
        """Download all models"""
        models = ["yolov8n", "lprnet", "vehicle_tracking"]
        success_count = 0
        
        for model_name in models:
            if self.download_model(model_name):
                success_count += 1
        
        console.print(f"[green]✅ Downloaded {success_count}/{len(models)} models successfully[/green]")
        return success_count == len(models)

def main():
    parser = argparse.ArgumentParser(description="OpenVINO AI Model Testing Suite")
    parser.add_argument("--model", choices=["yolov8n", "lprnet", "vehicle_tracking", "all"], 
                       default="all", help="Model to test")
    parser.add_argument("--input", type=str, help="Input image or video path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--download-models", action="store_true", help="Download all models")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    tester = OpenVINOModelTester()
    
    if args.list_models:
        # List available models
        table = Table(title="Available Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="green")
        
        models = ["yolov8n", "lprnet", "vehicle_tracking"]
        for model_name in models:
            try:
                config = tester.load_model_config(model_name)
                table.add_row(model_name, config["model_type"], config["description"])
            except FileNotFoundError:
                table.add_row(model_name, "Unknown", "Config not found")
        
        console.print(table)
        return
    
    if args.download_models:
        tester.download_all_models()
        return
    
    if not args.input:
        console.print("[red]❌ Please provide input path with --input[/red]")
        return
    
    if not os.path.exists(args.input):
        console.print(f"[red]❌ Input file not found: {args.input}[/red]")
        return
    
    # Run tests
    if args.model == "all":
        results = tester.run_all_tests(args.input)
        
        # Display results
        table = Table(title="Test Results")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        
        for model_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            table.add_row(model_name, status)
        
        console.print(table)
    else:
        # Test specific model
        is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        if is_video:
            # Video processing
            output_path = args.output or f"output/{args.model}_video_output.mp4"
            success = tester.test_video_processing(args.model, args.input, output_path)
        else:
            # Image processing
            if args.model == "yolov8n":
                success = tester.test_yolov8n_detection(args.input, args.output or "output/yolov8n_output.jpg")
            elif args.model == "lprnet":
                success = tester.test_lprnet_recognition(args.input, args.output or "output/lprnet_output.jpg")
            elif args.model == "vehicle_tracking":
                success = tester.test_vehicle_tracking(args.input, args.output or "output/vehicle_tracking_output.jpg")
        
        if success:
            console.print(f"[green]✅ {args.model} test completed successfully[/green]")
        else:
            console.print(f"[red]❌ {args.model} test failed[/red]")

if __name__ == "__main__":
    main()
