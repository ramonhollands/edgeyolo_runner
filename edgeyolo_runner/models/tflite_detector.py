import numpy as np
import cv2
from typing import List, Union, Tuple, Optional
import time

try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False


class TFLiteDetector:
    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.25,
        nms_thres: float = 0.5,
        fp16: bool = False,
        use_gpu: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """Initialize the TensorFlow Lite detector.
        
        Args:
            model_path: Path to the TFLite model file
            conf_thres: Confidence threshold for detections
            nms_thres: Non-maximum suppression threshold
            fp16: Whether to use FP16 precision
            use_gpu: Whether to use GPU acceleration (if available)
            input_size: Model input size (height, width). If None, will be extracted from model.
        """
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Please install it to use TFLiteDetector: pip install tensorflow")
        
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.fp16 = fp16

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        
        # Get GPU delegate if requested and available
        if use_gpu:
            try:
                # Try to use GPU delegate
                gpu_delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')
                self.interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[gpu_delegate]
                )
                print("Using GPU acceleration")
            except Exception as e:
                print(f"GPU acceleration not available, falling back to CPU: {e}")
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
        
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Extract input size from model if not provided
        if input_size is None:
            self.input_size = self._extract_input_size()
        else:
            self.input_size = input_size
        
        # Print model info
        print(f"TFLite Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        print(f"Input size: {self.input_size}")
        print(f"Using GPU: {use_gpu}")
        print(f"Using FP16: {fp16}")

    def _extract_input_size(self) -> Tuple[int, int]:
        """Extract input size from TFLite model.
        
        Returns:
            Tuple of (height, width)
        """
        input_shape = self.input_details[0]['shape']
        print('input_shape', input_shape)
        # TFLite models typically have shape [batch, height, width, channels] (NHWC)
        if len(input_shape) == 4:
            if input_shape[3] == 3 or input_shape[3] == 1:  # NHWC format
                height, width = input_shape[1], input_shape[2]
            elif input_shape[1] == 3 or input_shape[1] == 1:  # NCHW format
                height, width = input_shape[2], input_shape[3]
            else:
                # Assume NHWC as default for TFLite
                height, width = input_shape[1], input_shape[2]
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        # Handle dynamic dimensions
        if height <= 0:
            height = 640  # default fallback
        if width <= 0:
            width = 640  # default fallback
            
        return (height, width)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess image for inference.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Preprocessed image and scale ratio
        """
        height, width = self.input_size
        
        # rotate image 90 degrees
        if img.shape[0] > img.shape[1]:
            self.is_rotated = True
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            self.is_rotated = False
            
        # Calculate scale ratio
        h, w = img.shape[:2]
        self.ratio_h = height / h
        self.ratio_w = width / w
        new_h, new_w = int(h * self.ratio_h), int(w * self.ratio_w)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create canvas and paste image
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        canvas[:new_h, :new_w, :] = resized
        
        # Normalize and transpose for TFLite (NHWC format)
        canvas = canvas
        canvas = np.expand_dims(canvas, 0)
        
        if self.fp16:
            canvas = canvas.astype(np.float16)
        else:
            canvas = canvas.astype(np.float32)
            
        return canvas

    def postprocess(self, outputs: np.ndarray) -> np.ndarray:
        """Postprocess raw model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed detections
        """
        # Remove batch dimension if present
        if len(outputs.shape) == 3:
            outputs = outputs[0]
            
        # If no detections, return None
        if outputs.shape[0] == 0:
            return None
            
        # Get boxes, scores, and class_ids
        boxes = outputs[..., :4]
        boxes[..., 0] *= self.input_size[1]
        boxes[..., 1] *= self.input_size[0]
        boxes[..., 2] *= self.input_size[1]
        boxes[..., 3] *= self.input_size[0]
        
        if self.is_rotated:
            xs = boxes[..., 0].copy()
            ys = boxes[..., 1].copy()
            widths = boxes[..., 2].copy()
            heights = boxes[..., 3].copy()

            boxes[..., 0] = ys
            boxes[..., 1] = self.input_size[1] - xs
            boxes[..., 2] = heights
            boxes[..., 3] = widths

        # Scale boxes back to original image size
        boxes[..., 0] /= self.ratio_h
        boxes[..., 1] /= self.ratio_w
        boxes[..., 2] /= self.ratio_h
        boxes[..., 3] /= self.ratio_w
        
        scores = outputs[..., 4]
        class_ids = outputs[..., 5]
        
        # Filter out low confidence detections
        valid_detections = scores > self.conf_thres
        
        if not np.any(valid_detections):
            return None
            
        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        class_ids = class_ids[valid_detections]
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_thres,
            self.nms_thres
        )
        
        if len(indices) == 0:
            return None
            
        return np.concatenate([
            boxes[indices],
            scores[indices].reshape(-1, 1),
            class_ids[indices].reshape(-1, 1)
        ], axis=1)

    def __call__(self, imgs: Union[np.ndarray, List[np.ndarray]]) -> List[Optional[np.ndarray]]:
        """Run inference on images.
        
        Args:
            imgs: Single image or list of images
            
        Returns:
            List of detections for each image
        """
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
            
        results = []
        for img in imgs:
            # Preprocess
            input_tensor = self.preprocess(img)
            
            # Run inference
            t0 = time.time()
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            outputs = self.interpreter.get_tensor(self.output_details[0]['index'])
            self.dt = time.time() - t0
            
            print("inference time in ms: ", self.dt*1000)
            print("inference time in seconds: ", self.dt)
            
            # Postprocess
            detections = self.postprocess(outputs)
            results.append(detections)
            
        return results[0] if len(results) == 1 else results 