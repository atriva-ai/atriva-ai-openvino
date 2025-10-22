import numpy as np
import cv2
# from openvino import AsyncInferQueue
from openvino.runtime import InferRequest
from app.models import model_manager

def letterbox(img, new_shape=(640,640), color=(114,114,114)):
    """Letterbox resizing to preserve aspect ratio"""
    h0, w0 = img.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w0, new_h / h0)
    resized_w, resized_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (resized_w, resized_h))
    dw = new_w - resized_w
    dh = new_h - resized_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)

def preprocess_image(image_bytes: bytes, input_shape: tuple):
    """Preprocess image for YOLOv8 inference using letterbox"""
    _, _, h, w = input_shape  # Extract height and width from model shape
    target_shape = (w, h)  # Model expects (width, height)

    # Convert bytes to NumPy array
    image_array = np.frombuffer(image_bytes, np.uint8)

    # Decode image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Store original shape for postprocessing
    original_shape = image.shape

    # Use letterbox resizing to preserve aspect ratio
    resized_image, ratio, pad = letterbox(image, new_shape=target_shape)
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized_image = rgb_image.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    input_data = np.transpose(normalized_image, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    
    print(f"Preprocessed input shape: {input_data.shape}")
    print(f"Original image shape: {original_shape}")
    print(f"Letterbox ratio: {ratio}, pad: {pad}")

    return input_data, original_shape, ratio, pad

def run_inference(input_data, compiled_model):
    """Runs inference on input data using OpenVINO model."""
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Debug: Check input shape before inference
    print(f"Input shape before inference: {input_data.shape}, expected: {input_layer.shape}")

    # Convert input to OpenVINO-compatible format
    input_tensor = np.array(input_data, dtype=np.float32)

    # Create inference request
    infer_request = compiled_model.create_infer_request()

    # Start inference synchronously
    infer_request.infer({input_layer: input_tensor})
    print("Inference completed...")

    # Get the output - YOLOv8 format: (1, 84, 8400) for NMS output
    output = infer_request.get_output_tensor(0).data
    print(f"Output shape after inference: {output.shape}")

    return output

def postprocess_detections(outputs, original_shape, ratio, pad, classes, confidence_threshold=0.25):
    """Postprocess YOLOv8 detections with proper coordinate transformation - matching test implementation"""
    detections = []
    
    # Get the output tensor
    if isinstance(outputs, dict):
        output_tensor = list(outputs.values())[0]
    else:
        output_tensor = outputs
    
    print(f"Processing output shape: {output_tensor.shape}")
    
    # Check if this is the NMS output format (like in the test) or raw YOLOv8 format
    if len(output_tensor.shape) == 3 and output_tensor.shape[1] == 6:
        # NMS output format: (1, 6, N) where each detection has [x1, y1, x2, y2, score, class_id]
        print("Processing NMS output format")
        for i in range(output_tensor.shape[2]):
            x1, y1, x2, y2, score, class_id = output_tensor[0, :, i]
            
            if score > confidence_threshold:
                # Undo letterbox transformation
                x1 = (x1 - pad[0]) / ratio
                y1 = (y1 - pad[1]) / ratio
                x2 = (x2 - pad[0]) / ratio
                y2 = (y2 - pad[1]) / ratio
                
                # Clip to image boundaries
                orig_h, orig_w = original_shape[:2]
                x1 = max(0, min(int(x1), orig_w - 1))
                y1 = max(0, min(int(y1), orig_h - 1))
                x2 = max(0, min(int(x2), orig_w - 1))
                y2 = max(0, min(int(y2), orig_h - 1))
                
                # Get class name
                if int(class_id) < len(classes):
                    class_name = classes[int(class_id)]
                else:
                    class_name = f"class_{int(class_id)}"
                
                detections.append({
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(score),
                    "bbox_xyxy": [x1, y1, x2, y2]
                })
    
    else:
        # Raw YOLOv8 format: (1, 84, 8400) or similar
        print("Processing raw YOLOv8 output format")
        # Process each detection
        for i in range(output_tensor.shape[2]):  # 8400 detections
            # Extract bbox coordinates (x_center, y_center, width, height)
            x_center = output_tensor[0, 0, i]
            y_center = output_tensor[0, 1, i]
            width = output_tensor[0, 2, i]
            height = output_tensor[0, 3, i]
            
            # Extract class scores
            class_scores = output_tensor[0, 4:, i]
            
            # Get the best class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > confidence_threshold:
                # Convert from center format to corner format
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                # Undo letterbox transformation
                x1 = (x1 - pad[0]) / ratio
                y1 = (y1 - pad[1]) / ratio
                x2 = (x2 - pad[0]) / ratio
                y2 = (y2 - pad[1]) / ratio
                
                # Clip to image boundaries
                orig_h, orig_w = original_shape[:2]
                x1 = max(0, min(int(x1), orig_w - 1))
                y1 = max(0, min(int(y1), orig_h - 1))
                x2 = max(0, min(int(x2), orig_w - 1))
                y2 = max(0, min(int(y2), orig_h - 1))
                
                # Get class name
                if class_id < len(classes):
                    class_name = classes[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                detections.append({
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "bbox_xyxy": [x1, y1, x2, y2]
                })
    
    print(f"Found {len(detections)} detections")
    return detections

def run_object_detection(image_bytes: bytes, object_name: str):
    """Runs YOLOv8 inference using a dynamically selected model."""
    compiled_model, input_shape = model_manager.load_model(object_name)  # Load requested model by object name like "yolov8n"
    
    # Get model configuration for classes and thresholds
    model_config = model_manager.get_model_config(object_name)
    classes = model_config.get("classes", [])
    confidence_threshold = model_config.get("confidence_threshold", 0.25)
    
    # Preprocess image for model input (returns additional info for postprocessing)
    image_data, original_shape, ratio, pad = preprocess_image(image_bytes, input_shape)

    # Run inference
    model_output = run_inference(image_data, compiled_model)

    # Postprocess detections with proper coordinate transformation
    detections = postprocess_detections(
        model_output, 
        original_shape, 
        ratio, 
        pad, 
        classes, 
        confidence_threshold
    )

    # Print results
    for det in detections:
        print(f"Class: {det['class_name']} (ID: {det['class_id']}), Confidence: {det['confidence']:.3f}, BBox: {det['bbox_xyxy']}")

    return detections