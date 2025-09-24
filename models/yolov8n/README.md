# YOLOv8n Model Setup

## Current Status

✅ **Downloaded**: `yolov8n.pt` (6.5MB PyTorch model)  
⚠️ **Conversion Issue**: PyTorch doesn't support Python 3.13 yet

## Solution Options

### Option 1: Use Python 3.11 or 3.12 (Recommended)

Create a new virtual environment with Python 3.11 or 3.12:

```bash
# Using pyenv (if available)
pyenv install 3.11.9
pyenv virtualenv 3.11.9 openvino-conversion
pyenv activate openvino-conversion

# Or using conda
conda create -n openvino-conversion python=3.11
conda activate openvino-conversion

# Install required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
pip install openvino

# Convert the model
cd /path/to/models/yolov8n
python convert_to_openvino.py
```

### Option 2: Manual Conversion Using Docker

```bash
# Create a Docker container with Python 3.11
docker run -it --rm -v $(pwd):/workspace python:3.11 bash

# Inside the container:
cd /workspace
pip install torch torchvision ultralytics openvino
python convert_to_openvino.py
```

### Option 3: Use Pre-converted Models

Download from a working source:

```bash
# Try these alternative URLs:
curl -L -o yolov8n.onnx "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx"

# Then convert ONNX to OpenVINO using system tools:
mo --input_model yolov8n.onnx --output_dir . --model_name yolov8n
```

### Option 4: Temporary Mock Model (For Testing Only)

If you just need to test the infrastructure, you can create minimal mock files:

```bash
# Create minimal XML file (not a real model)
echo '<?xml version="1.0" ?><net><layers></layers></net>' > yolov8n.xml
echo 'mock' > yolov8n.bin
```

## Files in this directory

- `yolov8n.pt` - Downloaded PyTorch model (6.5MB)
- `model.json` - Model configuration
- `convert_to_openvino.py` - Conversion script (requires PyTorch)
- `download_and_convert.py` - Download and conversion script

## Next Steps

1. Choose one of the options above
2. Convert the PyTorch model to OpenVINO format
3. Test the model loading in the test suite

The conversion should create:
- `yolov8n.xml` - OpenVINO model structure
- `yolov8n.bin` - OpenVINO model weights
