import cv2
import numpy as np
import argparse
import os
import sys
from edgeyolo_runner.models import DetectorFactory

def draw_detections(image, detections, class_names=None):
    """Draw detection boxes on image."""
    if detections is None:
        return image
        
    image = image.copy()
    for det in detections:
        box = det[:4].astype(int)
        score = det[4]
        class_id = int(det[5])
        
        # Draw box
        box[0] = box[0] - box[2] / 2
        box[1] = box[1] - box[3] / 2
        cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (box[0], box[1] - label_height - baseline),
                     (box[0] + label_width, box[1]), (0, 255, 0), -1)
        cv2.putText(image, label, (box[0], box[1] - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def detect_model_type(model_path):
    """Detect model type based on file extension."""
    model_path = model_path.lower()
    if model_path.endswith('.onnx'):
        return 'onnx'
    elif model_path.endswith('.tflite'):
        return 'tflite'
    elif model_path.endswith(('.mlmodel', '.mlpackage')):
        return 'coreml'
    else:
        return 'auto'  # Let the factory try to auto-detect

def main():
    parser = argparse.ArgumentParser(description='EdgeYOLO Image Detection')
    parser.add_argument('--model', '-m', required=True, 
                       help='Path to model file (.onnx, .tflite, .mlmodel, .mlpackage)')
    parser.add_argument('--image', '-i', required=True, 
                       help='Path to input image')
    parser.add_argument('--output', '-o', 
                       help='Path to save output image (optional)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--nms-thres', type=float, default=0.5,
                       help='NMS threshold (default: 0.5)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 precision')
    parser.add_argument('--no-acceleration', action='store_true',
                       help='Disable hardware acceleration')
    parser.add_argument('--input-size', nargs=2, type=int, 
                       help='Model input size (height width). If not specified, will be extracted automatically from model.')
    parser.add_argument('--detector-type', choices=['onnx', 'tflite', 'coreml', 'auto'],
                       help='Force specific detector type (overrides auto-detection)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Determine detector type
    detector_type = args.detector_type if args.detector_type else detect_model_type(args.model)
    print(f"Detected model type: {detector_type}")
    
    try:
        # Initialize detector using factory
        input_size = tuple(args.input_size) if args.input_size else None
        detector = DetectorFactory.create_detector(
            model_path=args.model,
            detector_type=detector_type,
            conf_thres=args.conf_thres,
            nms_thres=args.nms_thres,
            fp16=args.fp16,
            use_acceleration=not args.no_acceleration,
            input_size=input_size
        )
        if input_size is None:
            print(f"✓ Successfully loaded {detector_type.upper()} detector with auto-detected input size: {detector.input_size}")
        else:
            print(f"✓ Successfully loaded {detector_type.upper()} detector with manual input size: {detector.input_size}")
        
    except Exception as e:
        print(f"Error loading detector: {e}")
        print("\nTroubleshooting:")
        if detector_type == 'tflite':
            print("- For TFLite models, install tensorflow: pip install tensorflow")
        elif detector_type == 'coreml':
            print("- For CoreML models, install coremltools: pip install coremltools")
        elif detector_type == 'onnx':
            print("- For ONNX models with GPU, install onnxruntime-gpu: pip install onnxruntime-gpu")
        sys.exit(1)
    
    # Load and process image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        sys.exit(1)
    
    print(f"Processing image: {args.image} ({image.shape[1]}x{image.shape[0]})")
    
    # Run detection
    detections = detector(image)
    
    # Print results
    if detections is not None:
        print(f"Found {len(detections)} detections")
        for i, det in enumerate(detections):
            box = det[:4]
            score = det[4]
            class_id = int(det[5])
            print(f"  Detection {i+1}: class={class_id}, conf={score:.3f}, box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    else:
        print("No detections found")
    
    # Draw results
    result = draw_detections(image, detections)

    print(f"Inference time: {detector.dt*1000:.2f}ms")
    
    # Save output if specified
    if args.output:
        cv2.imwrite(args.output, result)
        print(f"Output saved to: {args.output}")
    
    # Display results
    cv2.imshow("Detections", result)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 