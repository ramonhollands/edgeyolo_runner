# EdgeYOLO Runner

A Python package for running EdgeYOLO models using ONNX Runtime, TensorFlow Lite, and CoreML on images and videos. This package supports both FP32 and FP16 precision, and can run on both CPU and GPU acceleration.

## Installation

### Basic Installation (ONNX support only)
```bash
pip install edgeyolo_runner
```

### With TensorFlow Lite Support
```bash
pip install edgeyolo_runner[tflite]
# or for all optional dependencies
pip install edgeyolo_runner[all]
```

## Usage

### Using the DetectorFactory (Recommended)

The `DetectorFactory` automatically detects the model type based on the file extension:

```python
from edgeyolo_runner import DetectorFactory
import cv2

# Auto-detect model type (.onnx or .tflite)
detector = DetectorFactory.create_detector(
    model_path="path/to/your/model.onnx",  # or .tflite
    detector_type="auto",  # Auto-detect based on file extension
    conf_thres=0.25,
    nms_thres=0.5,
    use_acceleration=True  # CUDA for ONNX, GPU for TFLite
)

# Read and process image
image = cv2.imread("image.jpg")
detections = detector(image)

# Process detections
if detections is not None:
    for det in detections:
        boxes = det[:4]
        score = det[4]
        class_id = int(det[5])
        # Draw boxes, etc.
```

### ONNX Detector

```python
from edgeyolo_runner import ONNXDetector
import cv2

# Initialize ONNX detector
detector = ONNXDetector(
    model_path="path/to/your/model.onnx",
    conf_thres=0.25,
    nms_thres=0.5,
    fp16=False,  # Set to True for FP16 inference
    use_cuda=True  # Set to False for CPU inference
)

# Read and process image
image = cv2.imread("image.jpg")
detections = detector(image)
```

### TensorFlow Lite Detector

```python
from edgeyolo_runner import TFLiteDetector
import cv2

# Initialize TFLite detector
detector = TFLiteDetector(
    model_path="path/to/your/model.tflite",
    conf_thres=0.25,
    nms_thres=0.5,
    fp16=False,  # Set to True for FP16 inference
    use_gpu=True  # Set to False for CPU inference
)

# Read and process image
image = cv2.imread("image.jpg")
detections = detector(image)
```

### Video Detection

```python
from edgeyolo_runner import DetectorFactory
import cv2

# Works with both ONNX and TFLite models
detector = DetectorFactory.create_detector("path/to/your/model.onnx")

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    detections = detector(frame)
    # Process detections and draw on frame
    
    print(f"Inference time: {detector.dt*1000:.2f}ms")
    
cap.release()
```

### Command Line Usage

Use the provided example script to compare detectors:

```bash
# Auto-detect model type and run inference
python examples/detector_comparison.py --model model.onnx --image image.jpg

# Specifically use TFLite detector
python examples/detector_comparison.py --model model.tflite --image image.jpg --detector tflite

# Use hardware acceleration
python examples/detector_comparison.py --model model.onnx --image image.jpg --use-acceleration

# Save output
python examples/detector_comparison.py --model model.onnx --image image.jpg --output result.jpg
```

## Features

- **Dual Runtime Support**: Both ONNX Runtime and TensorFlow Lite
- **Auto Model Detection**: Automatic model type detection based on file extension
- **Hardware Acceleration**: CUDA support for ONNX, GPU delegation for TFLite
- **Precision Options**: FP16 and FP32 precision support
- **Easy-to-use API**: Unified interface for both model types
- **Batch Processing**: Support for processing multiple images
- **Performance Monitoring**: Built-in inference time measurement

## Model Support

- **ONNX Models**: `.onnx` files using ONNX Runtime
- **TensorFlow Lite Models**: `.tflite` files using TensorFlow Lite

## Requirements

### Core Dependencies
- Python >= 3.7
- ONNX Runtime >= 1.15.0
- OpenCV >= 4.5.0
- NumPy >= 1.19.0
- Pillow >= 8.0.0

### Optional Dependencies
- **TensorFlow >= 2.8.0**: For TensorFlow Lite support
- **ONNX Runtime GPU**: For CUDA acceleration

## License

MIT License 