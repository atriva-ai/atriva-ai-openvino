#!/usr/bin/env python3
"""
Vehicle Detection and Tracking Test
===================================

This script tests the vehicle detection and tracking model with OpenVINO.
It can process both images and videos, detecting and tracking vehicles with annotated outputs.

Usage:
    python test_vehicle_tracking.py --input test_images/sample_vehicles.jpg
    python test_vehicle_tracking.py --input test_videos/sample_traffic.mp4 --video
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
from collections import defaultdict

console = Console()

class VehicleTrackingTester:
    """Vehicle Detection and Tracking Tester"""
    
    def __init__(self):
        self.models_dir = Path("../models/vehicle_tracking")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model configuration
        with open(self.models_dir / "model.json", 'r') as f:
            self.config = json.load(f)
        
        # Initialize OpenVINO
        self.ie = Core()
        self.model = None
        self.input_shape = self.config["input_shape"][2:]  # [416, 416]
        self.classes = self.config["classes"]
        self.confidence_threshold = self.config["confidence_threshold"]
        self.nms_threshold = self.config["nms_threshold"]
        self.tracking_threshold = self.config["tracking_threshold"]
        self.max_tracks = self.config["max_tracks"]
        
        # Tracking state
        self.tracks = {}
        self.next_track_id = 1
        
        console.print(f"[green]Vehicle Tracking Tester initialized[/green]")
        console.print(f"[blue]Input shape: {self.input_shape}[/blue]")
        console.print(f"[blue]Vehicle classes: {self.classes}[/blue]")
        console.print(f"[blue]Max tracks: {self.max_tracks}[/blue]")
    
    def load_model(self):
        """Load Vehicle Detection OpenVINO model"""
        xml_path = self.models_dir / "vehicle_tracking.xml"
        bin_path = self.models_dir / "vehicle_tracking.bin"
        
        if not xml_path.exists() or not bin_path.exists():
            console.print("[red]Model files not found. Please download the model first.[/red]")
            return False
        
        try:
            model = self.ie.read_model(model=xml_path)
            self.model = self.ie.compile_model(model=model, device_name="CPU")
            console.print("[green]✅ Vehicle detection model loaded successfully[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {str(e)}[/red]")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for vehicle detection inference"""
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
        """Post-process vehicle detection results (simplified version)"""
        # This is a simplified post-processing
        # Real implementation would include proper NMS and coordinate transformation
        
        detections = []
        output_shape = outputs.shape  # [1, 255, 13, 13]
        
        # Extract detections (simplified)
        for i in range(output_shape[2]):  # 13x13 grid
            for j in range(output_shape[3]):
                confidence = outputs[0, 4, i, j]  # Objectness score
                
                if confidence > self.confidence_threshold:
                    # Get class probabilities
                    class_scores = outputs[0, 5:, i, j]
                    class_id = np.argmax(class_scores)
                    class_confidence = class_scores[class_id]
                    
                    if class_confidence > self.confidence_threshold:
                        # Get bounding box coordinates (simplified)
                        x_center = outputs[0, 0, i, j]
                        y_center = outputs[0, 1, i, j]
                        width = outputs[0, 2, i, j]
                        height = outputs[0, 3, i, j]
                        
                        detections.append({
                            'class_id': int(class_id),
                            'class_name': self.classes[class_id],
                            'confidence': float(class_confidence),
                            'bbox': [x_center, y_center, width, height]
                        })
        
        return detections
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_tracks(self, detections):
        """Update vehicle tracks based on new detections"""
        # Simple tracking algorithm using IoU matching
        matched_detections = set()
        updated_tracks = {}
        
        # Match detections to existing tracks
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_detection_idx = -1
            
            for i, detection in enumerate(detections):
                if i in matched_detections:
                    continue
                
                iou = self.calculate_iou(track['bbox'], detection['bbox'])
                if iou > best_iou and iou > self.tracking_threshold:
                    best_iou = iou
                    best_detection_idx = i
            
            if best_detection_idx != -1:
                # Update existing track
                detection = detections[best_detection_idx]
                updated_tracks[track_id] = {
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'age': track['age'] + 1,
                    'last_seen': 0
                }
                matched_detections.add(best_detection_idx)
            else:
                # Track not matched, increment age
                track['age'] += 1
                track['last_seen'] += 1
                
                # Remove old tracks
                if track['last_seen'] < 5:  # Keep tracks for 5 frames
                    updated_tracks[track_id] = track
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections and len(updated_tracks) < self.max_tracks:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                updated_tracks[track_id] = {
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'age': 1,
                    'last_seen': 0
                }
        
        self.tracks = updated_tracks
        return self.tracks
    
    def draw_tracks(self, image, tracks):
        """Draw vehicle tracks on image"""
        annotated_image = image.copy()
        
        for track_id, track in tracks.items():
            bbox = track['bbox']
            class_name = track['class_name']
            confidence = track['confidence']
            age = track['age']
            
            # Convert normalized coordinates to pixel coordinates
            h, w = image.shape[:2]
            x_center, y_center, width, height = bbox
            
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # Choose color based on track age
            if age > 10:
                color = (0, 255, 0)  # Green for stable tracks
            elif age > 5:
                color = (0, 255, 255)  # Yellow for medium tracks
            else:
                color = (0, 165, 255)  # Orange for new tracks
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and info
            label = f"ID:{track_id} {class_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_image
    
    def test_image(self, input_path, output_path):
        """Test vehicle detection and tracking on a single image"""
        console.print(f"[blue]Testing Vehicle Tracking on image: {input_path}[/blue]")
        
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
        
        # Update tracks
        tracks = self.update_tracks(detections)
        
        # Draw tracks
        annotated_image = self.draw_tracks(image, tracks)
        
        # Save result
        cv2.imwrite(output_path, annotated_image)
        
        console.print(f"[green]✅ Inference completed in {inference_time:.3f}s[/green]")
        console.print(f"[green]✅ Found {len(detections)} vehicles[/green]")
        console.print(f"[green]✅ Active tracks: {len(tracks)}[/green]")
        console.print(f"[green]✅ Saved annotated image: {output_path}[/green]")
        
        # Print detection details
        for i, detection in enumerate(detections):
            console.print(f"[cyan]  {i+1}. {detection['class_name']}: {detection['confidence']:.3f}[/cyan]")
        
        return True
    
    def test_video(self, input_path, output_path):
        """Test vehicle detection and tracking on a video"""
        console.print(f"[blue]Testing Vehicle Tracking on video: {input_path}[/blue]")
        
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
        max_tracks = 0
        
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
                
                # Update tracks
                tracks = self.update_tracks(detections)
                
                # Draw tracks
                annotated_frame = self.draw_tracks(frame, tracks)
                
                # Write frame
                out.write(annotated_frame)
                
                frame_count += 1
                total_detections += len(detections)
                max_tracks = max(max_tracks, len(tracks))
                progress.update(task, advance=1)
        
        cap.release()
        out.release()
        
        console.print(f"[green]✅ Processed {frame_count} frames[/green]")
        console.print(f"[green]✅ Total detections: {total_detections}[/green]")
        console.print(f"[green]✅ Average detections per frame: {total_detections/frame_count:.2f}[/green]")
        console.print(f"[green]✅ Maximum active tracks: {max_tracks}[/green]")
        console.print(f"[green]✅ Saved annotated video: {output_path}[/green]")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection and Tracking Test")
    parser.add_argument("--input", type=str, required=True, help="Input image or video path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--video", action="store_true", help="Process as video")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = VehicleTrackingTester()
    
    # Load model
    if not tester.load_model():
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_path = "output/vehicle_tracking_video_output.mp4"
        else:
            output_path = "output/vehicle_tracking_image_output.jpg"
    
    # Run test
    if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        success = tester.test_video(args.input, output_path)
    else:
        success = tester.test_image(args.input, output_path)
    
    if success:
        console.print("[green]🎉 Vehicle tracking test completed successfully![/green]")
    else:
        console.print("[red]❌ Vehicle tracking test failed![/red]")

if __name__ == "__main__":
    main()
