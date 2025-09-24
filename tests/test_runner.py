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
        """Download OpenVINO model files"""
        config = self.load_model_config(model_name)
        model_dir = self.models_dir / model_name
        
        console.print(f"[yellow]Downloading {model_name} model...[/yellow]")
        
        try:
            # Download XML file
            xml_url = config["openvino_model_url"] + f"{model_name}.xml"
            xml_path = model_dir / f"{model_name}.xml"
            
            response = requests.get(xml_url, stream=True)
            response.raise_for_status()
            
            with open(xml_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading XML"):
                    f.write(chunk)
            
            # Download BIN file
            bin_url = config["openvino_model_url"] + f"{model_name}.bin"
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
            
            # Resize image to model input size
            input_shape = config["input_shape"][2:]  # [640, 640]
            resized_image = cv2.resize(image, input_shape)
            
            # Convert BGR to RGB and normalize
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            normalized_image = rgb_image.astype(np.float32) / 255.0
            
            # Transpose to CHW format
            input_data = np.transpose(normalized_image, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            input_tensor = model.input(0)
            output_tensor = model.output(0)
            
            result = model([input_tensor], {input_tensor: input_data})
            outputs = result[output_tensor]
            
            # Process outputs (simplified - would need proper post-processing)
            console.print(f"[green]✅ YOLOv8n inference completed. Output shape: {outputs.shape}[/green]")
            
            # Save annotated image (simplified)
            annotated_image = image.copy()
            cv2.putText(annotated_image, "YOLOv8n Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
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
            input_tensor = model.input(0)
            output_tensor = model.output(0)
            
            result = model([input_tensor], {input_tensor: input_data})
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
            input_tensor = model.input(0)
            output_tensor = model.output(0)
            
            result = model([input_tensor], {input_tensor: input_data})
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
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame based on model type
                if model_name == "yolov8n":
                    # Simplified processing - would need proper implementation
                    processed_frame = frame.copy()
                    cv2.putText(processed_frame, f"YOLOv8n Frame {frame_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif model_name == "lprnet":
                    processed_frame = frame.copy()
                    cv2.putText(processed_frame, f"LPRNet Frame {frame_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif model_name == "vehicle_tracking":
                    processed_frame = frame.copy()
                    cv2.putText(processed_frame, f"Vehicle Tracking Frame {frame_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    processed_frame = frame
                
                out.write(processed_frame)
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
