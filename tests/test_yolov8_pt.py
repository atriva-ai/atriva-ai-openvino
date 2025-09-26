#!/usr/bin/env python3
"""
YOLOv8 PyTorch Object Detection Test
====================================

This script tests YOLOv8 object detection models (n/s/m) with PyTorch.
It can process both images and videos, detecting objects and creating annotated outputs.

Usage:
    python test_yolov8_pt.py --input test_images/sample.jpg
    python test_yolov8_pt.py --input test_images/sample.jpg --size s
    python test_yolov8_pt.py --input test_videos/sample.mp4 --video --size m
    python test_yolov8_pt.py --input test_videos/sample.mp4 --video --fps 30
    python test_yolov8_pt.py --input test_videos/sample.mp4 --video --length 10.5
    python test_yolov8_pt.py --input test_videos/sample.mp4 --video --inference-fps 1 --length 30
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import time

console = Console()

class YOLOv8PyTorchTester:
    """YOLOv8 PyTorch Object Detection Tester"""
    
    def __init__(self, model_size='n'):
        self.model_size = model_size
        self.model = None
        self.models_dir = Path(f"../models/yolov8{model_size}")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        console.print(f"[green]YOLOv8{model_size.upper()} PyTorch Tester initialized[/green]")
    
    def load_model(self):
        """Load YOLOv8 PyTorch model"""
        model_path = self.models_dir / f"yolov8{self.model_size}.pt"
        
        try:
            self.model = YOLO(model_path)
            console.print(f"[green]✅ YOLOv8{self.model_size.upper()} PyTorch model loaded successfully[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {str(e)}[/red]")
            return False
    
    def test_image(self, input_path, output_path):
        """Test YOLOv8 on a single image"""
        console.print(f"[blue]Testing YOLOv8{self.model_size.upper()} on image: {input_path}[/blue]")
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            console.print(f"[red]❌ Could not load image: {input_path}[/red]")
            return False
        
        # Run prediction
        start_time = time.time()
        results = self.model.predict(input_path, imgsz=640, conf=0.25, save=False)
        inference_time = time.time() - start_time
        
        # Process results
        detections = []
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    detections.append({
                        "class_id": int(cls_id),
                        "class_name": self.model.names[int(cls_id)],
                        "confidence": float(conf),
                        "bbox_xyxy": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    })
        
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
    
    def test_video(self, input_path, output_path, fps=None, inference_fps=None, length=None):
        """Test YOLOv8 on a video"""
        console.print(f"[blue]Testing YOLOv8{self.model_size.upper()} on video: {input_path}[/blue]")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            console.print(f"[red]❌ Could not open video: {input_path}[/red]")
            return False
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use provided fps or original video fps
        output_fps = fps if fps is not None else original_fps
        
        # Calculate frames to process based on length and inference FPS
        if length is not None:
            if inference_fps is not None:
                # Process only N frames per second for specified duration
                frames_to_process = int(length * inference_fps)
                total_duration = total_frames / original_fps
                console.print(f"[blue]Video properties: {width}x{height}, {original_fps} FPS (original), {output_fps} FPS (output)[/blue]")
                console.print(f"[blue]Processing {length}s at {inference_fps} FPS = {frames_to_process} frames out of {total_duration:.1f}s total[/blue]")
            else:
                # Process all frames for specified duration
                frames_to_process = int(length * original_fps)
                total_duration = total_frames / original_fps
                console.print(f"[blue]Video properties: {width}x{height}, {original_fps} FPS (original), {output_fps} FPS (output)[/blue]")
                console.print(f"[blue]Processing {length}s out of {total_duration:.1f}s total ({frames_to_process} frames)[/blue]")
        else:
            frames_to_process = total_frames
            console.print(f"[blue]Video properties: {width}x{height}, {original_fps} FPS (original), {output_fps} FPS (output), {total_frames} frames[/blue]")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        total_detections = 0
        
        # Calculate frame skip interval for inference FPS
        if inference_fps is not None:
            frame_skip = int(original_fps / inference_fps)
            console.print(f"[blue]Inference FPS: {inference_fps}, skipping {frame_skip-1} frames between inferences[/blue]")
        else:
            frame_skip = 1
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing video...", total=frames_to_process)
            
            while processed_frames < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on inference FPS
                if frame_count % frame_skip == 0:
                    # Run prediction on frame
                    results = self.model.predict(frame, imgsz=640, conf=0.25, save=False, verbose=False)
                    
                    # Process results
                    detections = []
                    for r in results:
                        if r.boxes is not None:
                            boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                            confidences = r.boxes.conf.cpu().numpy()
                            class_ids = r.boxes.cls.cpu().numpy().astype(int)
                            
                            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                                detections.append({
                                    "class_id": int(cls_id),
                                    "class_name": self.model.names[int(cls_id)],
                                    "confidence": float(conf),
                                    "bbox_xyxy": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                                })
                    
                    # Draw detections
                    annotated_frame = self.draw_detections(frame, detections)
                    
                    # Write frame
                    out.write(annotated_frame)
                    
                    processed_frames += 1
                    total_detections += len(detections)
                    progress.update(task, advance=1)
                
                frame_count += 1
        
        cap.release()
        out.release()
        
        console.print(f"[green]✅ Processed {processed_frames} frames (out of {frame_count} total frames)[/green]")
        console.print(f"[green]✅ Total detections: {total_detections}[/green]")
        console.print(f"[green]✅ Average detections per frame: {total_detections/processed_frames:.2f}[/green]")
        console.print(f"[green]✅ Saved annotated video: {output_path}[/green]")
        
        return True
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        annotated_image = image.copy()
        
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Use the xyxy format
            x1, y1, x2, y2 = detection['bbox_xyxy']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
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

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 PyTorch Object Detection Test")
    parser.add_argument("--input", type=str, required=True, help="Input image or video path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--video", action="store_true", help="Process as video")
    parser.add_argument("--fps", type=int, help="Output video FPS (overrides original video FPS)")
    parser.add_argument("--inference-fps", type=float, help="Inference FPS - process only N frames per second (default: process all frames)")
    parser.add_argument("--length", type=float, help="Number of seconds to process from video (default: process entire video)")
    parser.add_argument("--size", choices=['n', 's', 'm'], default='n', help="YOLOv8 model size: n (nano), s (small), m (medium)")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = YOLOv8PyTorchTester(model_size=args.size)
    
    # Load model
    if not tester.load_model():
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_path = f"output/yolov8{args.size}_pt_video_output.mp4"
        else:
            output_path = f"output/yolov8{args.size}_pt_image_output.jpg"
    
    # Run test
    if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        success = tester.test_video(args.input, output_path, args.fps, args.inference_fps, args.length)
    else:
        success = tester.test_image(args.input, output_path)
    
    if success:
        console.print(f"[green]🎉 YOLOv8{args.size.upper()} PyTorch test completed successfully![/green]")
    else:
        console.print(f"[red]❌ YOLOv8{args.size.upper()} PyTorch test failed![/red]")

if __name__ == "__main__":
    main()
