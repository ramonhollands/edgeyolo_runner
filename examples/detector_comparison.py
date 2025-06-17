import cv2
import numpy as np
import argparse
from edgeyolo_runner import ONNXDetector, TFLiteDetector, DetectorFactory

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
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (box[0], box[1] - label_height - baseline),
                     (box[0] + label_width, box[1]), (0, 255, 0), -1)
        cv2.putText(image, label, (box[0], box[1] - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def run_onnx_detector(model_path, image_path, use_cuda=False):
    """Run detection using ONNX detector."""
    print("\n=== ONNX Detector ===")
    
    # Initialize ONNX detector
    detector = ONNXDetector(
        model_path=model_path,
        conf_thres=0.25,
        nms_thres=0.5,
        fp16=False,
        use_cuda=use_cuda
    )
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
        
    # Run detection
    detections = detector(image)
    
    print(f"ONNX Inference time: {detector.dt*1000:.2f}ms")
    print(f"ONNX Detections: {len(detections) if detections is not None else 0}")
    
    return draw_detections(image, detections)

def run_tflite_detector(model_path, image_path, use_gpu=False):
    """Run detection using TFLite detector."""
    print("\n=== TFLite Detector ===")
    
    try:
        # Initialize TFLite detector
        detector = TFLiteDetector(
            model_path=model_path,
            conf_thres=0.25,
            nms_thres=0.5,
            fp16=False,
            use_gpu=use_gpu
        )
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None
            
        # Run detection
        detections = detector(image)
        
        print(f"TFLite Inference time: {detector.dt*1000:.2f}ms")
        print(f"TFLite Detections: {len(detections) if detections is not None else 0}")
        
        return draw_detections(image, detections)
        
    except ImportError as e:
        print(f"TFLite not available: {e}")
        return None

def run_factory_detector(model_path, image_path, use_acceleration=False):
    """Run detection using DetectorFactory (auto-detects model type)."""
    print("\n=== Factory Detector (Auto-detection) ===")
    
    try:
        # Initialize detector using factory
        detector = DetectorFactory.create_detector(
            model_path=model_path,
            detector_type="auto",  # Auto-detect based on file extension
            conf_thres=0.25,
            nms_thres=0.5,
            fp16=False,
            use_acceleration=use_acceleration
        )
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None
            
        # Run detection
        detections = detector(image)
        
        print(f"Factory Inference time: {detector.dt*1000:.2f}ms")
        print(f"Factory Detections: {len(detections) if detections is not None else 0}")
        
        return draw_detections(image, detections)
        
    except Exception as e:
        print(f"Factory detector error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare ONNX and TFLite detectors")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to model file (.onnx or .tflite)")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--detector", type=str, choices=["onnx", "tflite", "factory", "all"],
                       default="factory", help="Which detector to use")
    parser.add_argument("--use-acceleration", action="store_true",
                       help="Use hardware acceleration (CUDA for ONNX, GPU for TFLite)")
    parser.add_argument("--output", type=str, help="Output image path")
    
    args = parser.parse_args()
    
    result_image = None
    
    if args.detector == "onnx" or args.detector == "all":
        if args.model.lower().endswith('.onnx'):
            result_image = run_onnx_detector(args.model, args.image, args.use_acceleration)
        else:
            print("Error: ONNX detector requires .onnx model file")
    
    if args.detector == "tflite" or args.detector == "all":
        if args.model.lower().endswith('.tflite'):
            result_image = run_tflite_detector(args.model, args.image, args.use_acceleration)
        else:
            print("Error: TFLite detector requires .tflite model file")
    
    if args.detector == "factory":
        result_image = run_factory_detector(args.model, args.image, args.use_acceleration)
    
    # Save or display result
    if result_image is not None:
        if args.output:
            cv2.imwrite(args.output, result_image)
            print(f"Result saved to {args.output}")
        else:
            cv2.imshow("Detection Results", result_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 