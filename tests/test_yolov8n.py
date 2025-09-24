#!/usr/bin/env python3
"""
YOLOv8n Object Detection Test
============================

This script tests the YOLOv8n object detection model with OpenVINO.
It can process both images and videos, detecting objects and creating annotated outputs.

Usage:
    python test_yolov8n.py --input test_images/sample.jpg
    python test_yolov8n.py --input test_videos/sample.mp4 --video
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from openvino.runtime import Core
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import time

console = Console()

class YOLOv8nTester:
    """YOLOv8n Object Detection Tester"""
    
    def __init__(self):
        self.models_dir = Path("../models/yolov8n")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model configuration
        with open(self.models_dir / "model.json", 'r') as f:
            self.config = json.load(f)
        
        # Initialize OpenVINO
        self.ie = Core()
        self.model = None
        self.input_shape = self.config["input_shape"][2:]  # [640, 640]
        self.classes = self.config["classes"]
        self.confidence_threshold = self.config["confidence_threshold"]
        self.nms_threshold = self.config["nms_threshold"]
        
        console.print(f"[green]YOLOv8n Tester initialized[/green]")
        console.print(f"[blue]Input shape: {self.input_shape}[/blue]")
        console.print(f"[blue]Classes: {len(self.classes)}[/blue]")
    
    def load_model(self):
        """Load YOLOv8n OpenVINO model"""
        xml_path = self.models_dir / "yolov8n.xml"
        bin_path = self.models_dir / "yolov8n.bin"
        
        if not xml_path.exists() or not bin_path.exists():
            console.print("[red]Model files not found. Please download the model first.[/red]")
            return False
        
        try:
            model = self.ie.read_model(model=xml_path)
            self.model = self.ie.compile_model(model=model, device_name="CPU")
            console.print("[green]✅ YOLOv8n model loaded successfully[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {str(e)}[/red]")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for YOLOv8n inference"""
        # Resize image to model input size
        resized_image = cv2.resize(image, self.input_shape)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized_image = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        input_data = np.transpose(normalized_image, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    def postprocess_detections(self, outputs, original_shape):
        """Post-process YOLOv8n detections (simplified version)"""
        # This is a simplified post-processing
        # Real implementation would include proper NMS and coordinate transformation
        
        detections = []
        output_shape = outputs.shape  # [1, 84, 8400]
        
        # Extract detections (simplified)
        for i in range(output_shape[2]):  # 8400 detections
            confidence = outputs[0, 4, i]  # Objectness score
            
            if confidence > self.confidence_threshold:
                # Get class probabilities
                class_scores = outputs[0, 5:, i]
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                if class_confidence > self.confidence_threshold:
                    # Get bounding box coordinates (simplified)
                    x_center = outputs[0, 0, i]
                    y_center = outputs[0, 1, i]
                    width = outputs[0, 2, i]
                    height = outputs[0, 3, i]
                    
                    detections.append({
                        'class_id': int(class_id),
                        'class_name': self.classes[class_id],
                        'confidence': float(class_confidence),
                        'bbox': [x_center, y_center, width, height]
                    })
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        annotated_image = image.copy()
        
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            # Convert normalized coordinates to pixel coordinates
            h, w = image.shape[:2]
            x_center, y_center, width, height = bbox
            
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_image
    
    def test_image(self, input_path, output_path):
        """Test YOLOv8n on a single image"""
        console.print(f"[blue]Testing YOLOv8n on image: {input_path}[/blue]")
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            console.print(f"[red]❌ Could not load image: {input_path}[/red]")
            return False
        
        # Preprocess image
        input_data = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        input_tensor = self.model.input(0)
        output_tensor = self.model.output(0)
        
        result = self.model([input_tensor], {input_tensor: input_data})
        outputs = result[output_tensor]
        
        inference_time = time.time() - start_time
        
        # Post-process detections
        detections = self.postprocess_detections(outputs, image.shape)
        
        # Draw detections
        annotated_image = self.draw_detections(image, detections)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        
        # Print results
        console.print(f"[green]✅ Inference completed in {inference_time:.3f}s[/green]")
        console.print(f"[green]✅ Found {len(detections)} objects[/green]")
        console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
        
        # Print detection details
        for i, detection in enumerate(detections):
            console.print(f"[cyan]  {i+1}. {detection['class_name']}: {detection['confidence']:.3f}[/cyan]")
        
        return True
    
    def test_video(self, input_path, output_path):
        """Test YOLOv8n on a video"""
        console.print(f"[blue]Testing YOLOv8n on video: {input_path}[/blue]")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            console.print(f"[red]❌ Could not open video: {input_path}[/red]")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        console.print(f"[blue]Video properties: {width}x{height}, {fps} FPS, {total_frames} frames[/blue]")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing video...", total=total_frames)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                input_data = self.preprocess_image(frame)
                
                # Run inference
                input_tensor = self.model.input(0)
                output_tensor = self.model.output(0)
                
                result = self.model([input_tensor], {input_tensor: input_data})
                outputs = result[output_tensor]
                
                # Post-process detections
                detections = self.postprocess_detections(outputs, frame.shape)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Write frame
                out.write(annotated_frame)
                
                frame_count += 1
                total_detections += len(detections)
                progress.update(task, advance=1)
        
        cap.release()
        out.release()
        
        console.print(f"[green]✅ Processed {frame_count} frames[/green]")
        console.print(f"[green]✅ Total detections: {total_detections}[/green]")
        console.print(f"[green]✅ Average detections per frame: {total_detections/frame_count:.2f}[/green]")
        console.print(f"[green]✅ Saved annotated video: {output_path}[/green]")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n Object Detection Test")
    parser.add_argument("--input", type=str, required=True, help="Input image or video path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--video", action="store_true", help="Process as video")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = YOLOv8nTester()
    
    # Load model
    if not tester.load_model():
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_path = "output/yolov8n_video_output.mp4"
        else:
            output_path = "output/yolov8n_image_output.jpg"
    
    # Run test
    if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        success = tester.test_video(args.input, output_path)
    else:
        success = tester.test_image(args.input, output_path)
    
    if success:
        console.print("[green]🎉 YOLOv8n test completed successfully![/green]")
    else:
        console.print("[red]❌ YOLOv8n test failed![/red]")

if __name__ == "__main__":
    main()
