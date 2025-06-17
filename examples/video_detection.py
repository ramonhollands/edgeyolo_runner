import cv2
import numpy as np
import argparse
import os
import sys
import time
from edgeyolo_runner.models import DetectorFactory
from image_detection import draw_detections, detect_model_type

def process_video(detector, video_path, output_path=None, display=True, max_frames=None):
    """Process video with the given detector."""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Initialize video writer if output path is specified
    writer = None
    if output_path:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        print(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    total_time = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break
                
            # Check max frames limit
            if max_frames and frame_count >= max_frames:
                print(f"Reached maximum frames limit: {max_frames}")
                break
            
            # Run detection
            detections = detector(frame)
            
            # Draw results
            result = draw_detections(frame, detections)
            
            # Update statistics
            frame_count += 1
            total_time += detector.dt
            
            # Write frame
            if writer:
                writer.write(result)
                
            # Display frame
            if display:
                cv2.imshow("Video Detections", result)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User requested quit")
                    break
                elif key == ord(' '):
                    print("Paused - press any key to continue")
                    cv2.waitKey(0)
                    
            # Print progress
            if frame_count % 30 == 0 or frame_count == 1:
                elapsed_time = time.time() - start_time
                avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                inference_fps = frame_count / total_time if total_time > 0 else 0
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Frame {frame_count:5d}/{total_frames} ({progress:5.1f}%) | "
                      f"Processing FPS: {avg_fps:5.1f} | Inference FPS: {inference_fps:5.1f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    # Print final statistics
    elapsed_time = time.time() - start_time
    if frame_count > 0:
        avg_fps = frame_count / elapsed_time
        avg_inference_fps = frame_count / total_time
        print(f"\nFinal Statistics:")
        print(f"Processed {frame_count} frames in {elapsed_time:.2f}s")
        print(f"Average processing FPS: {avg_fps:.2f}")
        print(f"Average inference FPS: {avg_inference_fps:.2f}")
        print(f"Average inference time: {(total_time/frame_count)*1000:.2f}ms")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='EdgeYOLO Video Detection')
    parser.add_argument('--model', '-m', required=True, 
                       help='Path to model file (.onnx, .tflite, .mlmodel, .mlpackage)')
    parser.add_argument('--video', '-v', required=True, 
                       help='Path to input video file or camera index (0, 1, etc.)')
    parser.add_argument('--output', '-o', 
                       help='Path to save output video (optional)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--nms-thres', type=float, default=0.5,
                       help='NMS threshold (default: 0.5)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 precision')
    parser.add_argument('--no-acceleration', action='store_true',
                       help='Disable hardware acceleration')
    parser.add_argument('--detector-type', choices=['onnx', 'tflite', 'coreml', 'auto'],
                       help='Force specific detector type (overrides auto-detection)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display (useful for headless processing)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum number of frames to process (for testing)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Handle video input (file or camera)
    try:
        # Try to interpret as camera index
        camera_index = int(args.video)
        video_path = camera_index
        print(f"Using camera index: {camera_index}")
    except ValueError:
        # It's a file path
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
        print(f"Using video file: {video_path}")
    
    # Determine detector type
    detector_type = args.detector_type if args.detector_type else detect_model_type(args.model)
    print(f"Detected model type: {detector_type}")
    
    try:
        # Initialize detector using factory
        detector = DetectorFactory.create_detector(
            model_path=args.model,
            detector_type=detector_type,
            conf_thres=args.conf_thres,
            nms_thres=args.nms_thres,
            fp16=args.fp16,
            use_acceleration=not args.no_acceleration
        )
        print(f"✓ Successfully loaded {detector_type.upper()} detector")
        
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
    
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press SPACE to pause/resume")
    print("- Press any key to continue when paused")
    print()
    
    # Process video
    success = process_video(
        detector=detector,
        video_path=video_path,
        output_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames
    )
    
    if success:
        print("Video processing completed successfully!")
    else:
        print("Video processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 