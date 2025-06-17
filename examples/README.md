# EdgeYOLO Detection Examples

This directory contains examples for running EdgeYOLO models with automatic detector selection based on model file extensions.

## Features

- **Automatic Detector Selection**: Automatically detects and uses the correct detector (ONNX, TFLite, or CoreML) based on file extension
- **Automatic Input Size Extraction**: Automatically extracts input dimensions from model files - no need to specify manually
- **Multi-format Support**: Supports `.onnx`, `.tflite`, `.mlmodel`, and `.mlpackage` files
- **Command Line Interface**: Full CLI with configurable parameters
- **Hardware Acceleration**: Automatic hardware acceleration when available
- **Comprehensive Error Handling**: Clear error messages and troubleshooting hints

## Installation

### Basic Installation
```bash
pip install -e .
```

### With Optional Dependencies
```bash
# For TFLite support
pip install -e .[tflite]

# For CoreML support (macOS only)
pip install -e .[coreml]

# For GPU ONNX support
pip install -e .[gpu]

# For all features
pip install -e .[all]
```

## Examples

### 1. Image Detection (`image_detection.py`)

Detect objects in a single image.

#### Basic Usage
```bash
# Auto-detect model type and input size from model file
python examples/image_detection.py --model animal_detector_640_l.onnx --image Homosapiens.png

# Save output image
python examples/image_detection.py --model animal_detector_640_l.onnx --image Homosapiens.png --output result.jpg

# Use TFLite model with automatic input size extraction
python examples/image_detection.py --model animal_detector_float32.tflite --image Homosapiens.png

# Use CoreML model (macOS only)
python examples/image_detection.py --model animal_detector.mlmodel --image Homosapiens.png
```

#### Advanced Usage
```bash
# Custom thresholds and settings
python examples/image_detection.py \
    --model animal_detector_640_l.onnx \
    --image Homosapiens.png \
    --conf-thres 0.5 \
    --nms-thres 0.4 \
    --input-size 640 640 \
    --fp16

# Force specific detector type
python examples/image_detection.py \
    --model model.bin \
    --image test.jpg \
    --detector-type onnx

# Disable hardware acceleration
python examples/image_detection.py \
    --model animal_detector_640_l.onnx \
    --image Homosapiens.png \
    --no-acceleration
```

#### Arguments
- `--model, -m`: Path to model file (required)
- `--image, -i`: Path to input image (required)
- `--output, -o`: Path to save output image (optional)
- `--conf-thres`: Confidence threshold (default: 0.25)
- `--nms-thres`: NMS threshold (default: 0.5)
- `--input-size`: Model input size as height width (auto-detected from model if not specified)
- `--fp16`: Use FP16 precision
- `--no-acceleration`: Disable hardware acceleration
- `--detector-type`: Force specific detector type (auto, onnx, tflite, coreml)

### 2. Video Detection (`video_detection.py`)

Detect objects in video files or camera streams.

#### Basic Usage
```bash
# Process video file
python examples/video_detection.py --model lego_detector_128x256.onnx --video bricks.mp4

# Save processed video
python examples/video_detection.py \
    --model lego_detector_128x256.onnx \
    --video bricks.mp4 \
    --output output.mp4

# Use camera (index 0)
python examples/video_detection.py --model animal_detector_640_l.onnx --video 0
```

#### Advanced Usage
```bash
# Headless processing (no display)
python examples/video_detection.py \
    --model animal_detector_640_l.onnx \
    --video input.mp4 \
    --output result.mp4 \
    --no-display

# Process only first 100 frames
python examples/video_detection.py \
    --model animal_detector_640_l.onnx \
    --video input.mp4 \
    --max-frames 100

# Custom settings
python examples/video_detection.py \
    --model animal_detector_640_l.onnx \
    --video input.mp4 \
    --conf-thres 0.3 \
    --nms-thres 0.4 \
    --input-size 640 640 \
    --fp16
```

#### Video Controls
- Press `q` to quit
- Press `SPACE` to pause/resume
- Press any key to continue when paused

#### Arguments
All image detection arguments plus:
- `--video, -v`: Path to video file or camera index (required)
- `--no-display`: Disable video display (useful for headless processing)
- `--max-frames`: Maximum number of frames to process (for testing)

## Supported Model Formats

### ONNX (.onnx)
- **Hardware Acceleration**: CUDA GPU support
- **Installation**: `pip install onnxruntime` or `pip install onnxruntime-gpu`
- **Platforms**: Windows, Linux, macOS

### TensorFlow Lite (.tflite)
- **Hardware Acceleration**: GPU delegate support
- **Installation**: `pip install tensorflow`
- **Platforms**: Windows, Linux, macOS, mobile

### CoreML (.mlmodel, .mlpackage)
- **Hardware Acceleration**: Neural Engine, GPU support
- **Installation**: `pip install coremltools`
- **Platforms**: macOS only

## File Extension Auto-Detection

The examples automatically detect the model type based on file extension:

- `.onnx` → ONNX Runtime
- `.tflite` → TensorFlow Lite
- `.mlmodel` → CoreML
- `.mlpackage` → CoreML

You can override auto-detection using the `--detector-type` argument.

## Automatic Input Size Extraction

The examples automatically extract the input dimensions from model files, eliminating the need to manually specify `--input-size` in most cases.

### How it works

- **ONNX models**: Reads input tensor shape from model metadata
- **TFLite models**: Extracts input shape from model interpreter
- **CoreML models**: Reads input dimensions from model specification

### Example output
```bash
$ python examples/image_detection.py --model animal_detector_640_l.onnx --image test.jpg
Model loaded: animal_detector_640_l.onnx
Input name: images
Output name: output0
Input size: (640, 640)  # <- Automatically detected
✓ Successfully loaded ONNX detector with auto-detected input size: (640, 640)
```

### Manual override
You can still manually specify input size if needed:
```bash
python examples/image_detection.py --model model.onnx --image test.jpg --input-size 512 512
```

### Fallback behavior
If automatic extraction fails or encounters dynamic dimensions, the system falls back to (640, 640) as a default.

## Troubleshooting

### Import Errors
```bash
# For TFLite models
pip install tensorflow

# For CoreML models (macOS only)
pip install coremltools

# For ONNX GPU support
pip install onnxruntime-gpu
```

### Performance Issues
- Use `--fp16` for faster inference on supported hardware
- Use `--no-acceleration` if having hardware issues
- Adjust `--input-size` to match your model's requirements

### Memory Issues
- Use `--max-frames` to limit processing for testing
- Use `--no-display` for headless processing
- Use smaller input sizes if possible

## Example Model Downloads

The examples expect model files in the root directory. You can use your own models or train EdgeYOLO models following the main EdgeYOLO documentation.

## Creating Your Own Examples

You can use the detector factory in your own scripts:

```python
from edgeyolo_runner.models import DetectorFactory

# Auto-detect based on file extension
detector = DetectorFactory.create_detector(
    model_path="your_model.onnx",
    conf_thres=0.25,
    nms_thres=0.5
)

# Process image
import cv2
image = cv2.imread("your_image.jpg")
detections = detector(image)
``` 