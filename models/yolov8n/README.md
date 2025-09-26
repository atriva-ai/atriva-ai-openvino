# YOLOv8n Model Setup

## Model Sources

### Fine-tuned Models (Primary)
✅ **Using**: Fine-tuned YOLOv8n models from [shreydan/yolo-object-detection-kitti](https://github.com/shreydan/yolo-object-detection-kitti/tree/main)

These models are specifically fine-tuned for object detection tasks and provide better performance than the base Ultralytics models.

**Source**: [KITTI Object Detection with YOLOv8](https://github.com/shreydan/yolo-object-detection-kitti/tree/main)

### Original Ultralytics Models (Backup)
📁 **Location**: `ultralytics/` folder

The original Ultralytics YOLOv8 models are stored in the `ultralytics/` subfolder for reference and fallback purposes.

## Current Status

✅ **Downloaded**: `yolov8n.pt` (Fine-tuned PyTorch model)  
✅ **Converted**: `yolov8n.xml` and `yolov8n.bin` (OpenVINO format)  
✅ **Ready**: Model is ready for inference

## Model Files

### Primary Model Files
- `yolov8n.pt` - Fine-tuned PyTorch model
- `yolov8n.xml` - OpenVINO model structure
- `yolov8n.bin` - OpenVINO model weights
- `model.json` - Model configuration
- `metadata.yaml` - Model metadata

### Backup/Reference Files
- `ultralytics/` - Original Ultralytics models
  - `yolov8n.pt` - Original Ultralytics PyTorch model
  - `yolov8n.xml` - Original OpenVINO structure
  - `yolov8n.bin` - Original OpenVINO weights

## Code Organization Suggestion

⚠️ **Recommended**: Move Python scripts out of the model directory

The model directory should contain only model files. Consider moving Python scripts to:

```
scripts/
├── convert_to_openvino.py
├── download_and_convert.py
├── create_model_json.py
└── model_utils.py

models/
├── yolov8n/
│   ├── yolov8n.pt
│   ├── yolov8n.xml
│   ├── yolov8n.bin
│   ├── model.json
│   ├── metadata.yaml
│   └── ultralytics/
└── yolov8s/
    └── ...
```

## Usage

### Testing the Model
```bash
# Test with PyTorch
python tests/test_yolov8_pt.py --input test_images/sample.jpg --size n

# Test with OpenVINO
python tests/test_yolov8_openvino.py --input test_images/sample.jpg --size n
```

### Model Conversion (if needed)
```bash
# From scripts directory
python scripts/convert_to_openvino.py --size n
```

## Performance Benefits

The fine-tuned models from the KITTI dataset provide:
- Better accuracy on object detection tasks
- Optimized for real-world scenarios
- Improved performance on vehicle and pedestrian detection
- Enhanced robustness compared to base models

## References

- **Fine-tuned Models**: [shreydan/yolo-object-detection-kitti](https://github.com/shreydan/yolo-object-detection-kitti/tree/main)
- **Original YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **KITTI Dataset**: [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/index.php)
