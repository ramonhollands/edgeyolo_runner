#!/usr/bin/env python3
"""
TFLite Image Detection Example

This script demonstrates how to use the TFLiteDetector class for object detection
on images using TensorFlow Lite models.

Usage:
    python examples/image_detection_tflite.py --model path/to/model.tflite --image path/to/image.jpg
"""

import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time

# Add the parent directory to sys.path to import from models
sys.path.append(str(Path(__file__).parent.parent))

from edgeyolo_runner import TFLiteDetector

# COCO class names (you can modify this list based on your model's classes)
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def draw_detections(image: np.ndarray, detections: np.ndarray, class_names: list) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image: Input image
        detections: Detection results [x1, y1, x2, y2, confidence, class_id]
        class_names: List of class names
        
    Returns:
        Image with drawn detections
    """
    if detections is None:
        return image
        
    img_with_boxes = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)
        
        # Get class name
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )
    
    return img_with_boxes

def main():
    parser = argparse.ArgumentParser(description="TFLite Image Detection Example")
    parser.add_argument("--model", required=True, help="Path to TFLite model file")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save output image (optional)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms-thres", type=float, default=0.5, help="NMS threshold")
    parser.add_argument("--input-size", nargs=2, type=int, default=[320, 320], 
                       help="Model input size (height width)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration if available")
    parser.add_argument("--show", action="store_true", help="Show the result image")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return
    
    print("Initializing TFLite detector...")
    
    # Initialize detector
    try:
        detector = TFLiteDetector(
            model_path=args.model,
            conf_thres=args.conf_thres,
            nms_thres=args.nms_thres,
            fp16=args.fp16,
            use_gpu=args.use_gpu,
            input_size=tuple(args.input_size)
        )
        print("✓ Detector initialized successfully")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image '{args.image}'")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Run detection
    print("Running inference...")
    


    detections = detector(image)


    
    if detections is not None:
        print(f"Found {len(detections)} detections")
        
        # Print detection details
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, confidence, class_id = detection
            class_name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f"Class {int(class_id)}"
            print(f"Detection {i+1}: {class_name} ({confidence:.3f}) at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        # Draw detections on image
        result_image = draw_detections(image, detections, CLASS_NAMES)
        
        # Save output image if specified
        if args.output:
            cv2.imwrite(args.output, result_image)
            print(f"Output saved to: {args.output}")
        
        # Show image if requested
        if args.show:
            cv2.imshow("TFLite Detection Results", result_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No detections found")
        if args.show:
            cv2.imshow("TFLite Detection Results", image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
