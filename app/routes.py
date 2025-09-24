from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import run_object_detection
from app.models import model_manager, ACCELERATORS, MODEL_NAME_MAPPING
from app.model_capabilities import get_detailed_model_info, get_available_model_types, get_available_objects as get_capability_objects, get_model_capabilities
from app.shared_data import (
    list_available_cameras, 
    get_frame_info, 
    get_latest_frame, 
    read_frame_file,
    list_camera_frames
)
from fastapi.responses import FileResponse
import os
from threading import Thread
from typing import Optional

router = APIRouter()

ARCHITECTURE = "openvino"

@router.get("/models")
async def list_available_models():
    """Returns a list of all available models."""
    return {"available_models": model_manager.list_models()}

@router.get("/objects")
async def list_available_objects():
    """Returns a list of all available object types for detection."""
    return {"available_objects": get_capability_objects()}

@router.post("/inference/detection")
def detect_objects(object_name: str, image: UploadFile = File(...)):
    image_bytes = image.file.read()
    detections = run_object_detection(image_bytes, object_name)
    return {"objects": detections}

@router.get("/shared/cameras")
async def list_cameras():
    """List all cameras that have decoded frames available."""
    cameras = list_available_cameras()
    return {"cameras": cameras}

@router.get("/shared/cameras/{camera_id}/frames")
async def get_camera_frames(camera_id: str):
    """Get information about decoded frames for a specific camera."""
    frame_info = get_frame_info(camera_id)
    return frame_info

@router.get("/shared/cameras/{camera_id}/frames/latest")
async def get_camera_latest_frame(camera_id: str):
    """Get the latest decoded frame for a camera."""
    latest_frame = get_latest_frame(camera_id)
    if not latest_frame:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    return FileResponse(latest_frame)

@router.post("/shared/cameras/{camera_id}/inference")
async def detect_objects_in_camera_frame(camera_id: str, object_name: str):
    """Run object detection on the latest frame from a camera."""
    latest_frame = get_latest_frame(camera_id)
    if not latest_frame:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    # Read the frame file
    frame_bytes = read_frame_file(latest_frame)
    if not frame_bytes:
        raise HTTPException(status_code=500, detail=f"Failed to read frame file: {latest_frame}")
    
    # Run object detection
    detections = run_object_detection(frame_bytes, object_name)
    
    return {
        "camera_id": camera_id,
        "frame_path": latest_frame,
        "object_name": object_name,
        "detections": detections
    }

@router.get("/shared/cameras/{camera_id}/frames/{frame_index}")
async def get_camera_frame_by_index(camera_id: str, frame_index: int):
    """Get a specific frame by index for a camera."""
    frame_files = list_camera_frames(camera_id)
    if not frame_files:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    if frame_index < 0 or frame_index >= len(frame_files):
        raise HTTPException(
            status_code=400, 
            detail=f"Frame index {frame_index} out of range. Available frames: 0-{len(frame_files)-1}"
        )
    
    return FileResponse(frame_files[frame_index])

# --- Model Info API ---
@router.get("/model/info")
async def get_model_info():
    return {
        "models": get_detailed_model_info(),
        "model_types": get_available_model_types(),
        "objects": get_capability_objects(),  # Legacy compatibility
        "accelerators": ACCELERATORS,
        "architecture": ARCHITECTURE
    }

# --- Model Load API ---
@router.post("/model/load")
async def load_model(model_name: str, accelerator: Optional[str] = "cpu32"):
    if accelerator not in ACCELERATORS:
        raise HTTPException(status_code=400, detail=f"Unsupported accelerator: {accelerator}")
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    try:
        # Create a new manager for the requested accelerator
        from app.models import ModelManager
        manager = ModelManager(acceleration=accelerator)
        compiled_model, input_shape = manager.load_model(model_name)
        return {
            "model_name": model_name,
            "accelerator": accelerator,
            "architecture": ARCHITECTURE,
            "status": "loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Direct Inference API ---
@router.post("/inference/direct")
async def direct_inference(model_name: str, image: UploadFile = File(...)):
    """Run direct inference using a specific model."""
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    try:
        from app.services import preprocess_image, run_inference
        
        image_bytes = image.file.read()
        
        # Load model and get input shape
        compiled_model, input_shape = model_manager.load_model(model_name)
        
        # Preprocess image
        preprocessed_image = preprocess_image(image_bytes, input_shape)
        
        # Run inference
        output = run_inference(preprocessed_image, compiled_model)
        
        return {
            "model_name": model_name,
            "input_shape": input_shape,
            "output_shape": output.shape if hasattr(output, 'shape') else None,
            "output": output.tolist() if hasattr(output, 'tolist') else str(output)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Inference on Latest Frame API ---
@router.post("/inference/latest-frame")
async def inference_latest_frame(camera_id: str, model_name: str, accelerator: Optional[str] = "cpu32"):
    if accelerator not in ACCELERATORS:
        raise HTTPException(status_code=400, detail=f"Unsupported accelerator: {accelerator}")
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    latest_frame = get_latest_frame(camera_id)
    if not latest_frame:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    frame_bytes = read_frame_file(latest_frame)
    if not frame_bytes:
        raise HTTPException(status_code=500, detail=f"Failed to read frame file: {latest_frame}")
    from app.models import ModelManager
    manager = ModelManager(acceleration=accelerator)
    from app.services import preprocess_image, run_inference
    compiled_model, input_shape = manager.load_model(model_name)
    image = preprocess_image(frame_bytes, input_shape)
    model_output = run_inference(image, compiled_model)
    # Parse output (same as run_object_detection)
    confidence_threshold = 0.3
    detections = []
    for detection in model_output:
        image_id, class_id, confidence, xmin, ymin, xmax, ymax = detection
        if confidence > confidence_threshold:
            detections.append({
                "class_id": int(class_id),
                "confidence": float(confidence),
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
            })
    return {
        "camera_id": camera_id,
        "model_name": model_name,
        "accelerator": accelerator,
        "architecture": ARCHITECTURE,
        "frame_path": latest_frame,
        "detections": detections
    }

# --- Background Inference API ---
def background_inference(camera_id: str, model_name: str, accelerator: str):
    from app.models import ModelManager
    from app.services import preprocess_image, run_inference
    manager = ModelManager(acceleration=accelerator)
    compiled_model, input_shape = manager.load_model(model_name)
    frame_files = list_camera_frames(camera_id)
    results = []
    for frame_path in frame_files:
        frame_bytes = read_frame_file(frame_path)
        if not frame_bytes:
            continue
        image = preprocess_image(frame_bytes, input_shape)
        model_output = run_inference(image, compiled_model)
        confidence_threshold = 0.3
        detections = []
        for detection in model_output:
            image_id, class_id, confidence, xmin, ymin, xmax, ymax = detection
            if confidence > confidence_threshold:
                detections.append({
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                })
        results.append({
            "frame_path": frame_path,
            "detections": detections
        })
    # Optionally: save results to a file or database
    print(f"Background inference for camera {camera_id} complete. {len(results)} frames processed.")

@router.post("/inference/background")
async def start_background_inference(camera_id: str, model_name: str, accelerator: Optional[str] = "cpu32"):
    if accelerator not in ACCELERATORS:
        raise HTTPException(status_code=400, detail=f"Unsupported accelerator: {accelerator}")
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    thread = Thread(target=background_inference, args=(camera_id, model_name, accelerator), daemon=True)
    thread.start()
    return {
        "camera_id": camera_id,
        "model_name": model_name,
        "accelerator": accelerator,
        "architecture": ARCHITECTURE,
        "status": "background_inference_started"
    }

# --- Model Capabilities API ---
@router.get("/models/{model_name}/capabilities")
async def get_model_capabilities_endpoint(model_name: str):
    """Get detailed capabilities for a specific model."""
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    
    capabilities = get_model_capabilities(model_name)
    return {
        "model_name": model_name,
        "capabilities": capabilities,
        "architecture": ARCHITECTURE
    }

