import numpy as np
import onnxruntime as ort
import cv2
from typing import List, Union, Tuple, Optional
import time


class ONNXDetector:
    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.8,
        nms_thres: float = 0.5,
        fp16: bool = False,
        use_cuda: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """Initialize the ONNX detector.
        
        Args:
            model_path: Path to the ONNX model file
            conf_thres: Confidence threshold for detections
            nms_thres: Non-maximum suppression threshold
            fp16: Whether to use FP16 precision
            use_cuda: Whether to use CUDA GPU acceleration
            input_size: Model input size (height, width). If None, will be extracted from model.
        """
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.fp16 = fp16

        # Setup ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Extract input size from model if not provided
        if input_size is None:
            self.input_size = self._extract_input_size()
        else:
            self.input_size = input_size
        
        # Print model info
        print(f"Model loaded: {model_path}")
        print(f"Input name: {self.input_name}")
        print(f"Output name: {self.output_name}")
        print(f"Input size: {self.input_size}")
        print(f"Using CUDA: {use_cuda}")
        print(f"Using FP16: {fp16}")

    def _extract_input_size(self) -> Tuple[int, int]:
        """Extract input size from ONNX model.
        
        Returns:
            Tuple of (height, width)
        """

        
        input_shape = self.session.get_inputs()[0].shape
        # ONNX models typically have shape [batch, channels, height, width] or [batch, height, width, channels]
        if len(input_shape) == 4:
            # Assume NCHW format (batch, channels, height, width)
            if input_shape[1] == 3 or input_shape[1] == 1:  # channels first
                height, width = input_shape[2], input_shape[3]
            else:  # NHWC format (batch, height, width, channels)
                height, width = input_shape[1], input_shape[2]
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        # Handle dynamic dimensions (often marked as -1 or symbolic names)
        if isinstance(height, str) or height <= 0:
            height = 640  # default fallback
        if isinstance(width, str) or width <= 0:
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
            
        
        h, w = img.shape[:2]
        # ratio = min(height / h, width / w)
        self.ratio_h = height / h
        self.ratio_w = width / w
        new_h, new_w = int(h * self.ratio_h), int(w * self.ratio_w)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create canvas and paste image
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        canvas[:new_h, :new_w, :] = resized
        
        # Normalize and transpose
        canvas = canvas    
        canvas = canvas.transpose(2, 0, 1)
        # save canvas to file
        
        canvas = np.expand_dims(canvas, 0)
        
        if self.fp16:
            canvas = canvas.astype(np.float16)
            
        return canvas

    def postprocess(self, outputs: np.ndarray) -> np.ndarray:
        """Postprocess raw model outputs.
        
        Args:
            outputs: Raw model outputs
            ratio: Scale ratio from preprocessing
            
        Returns:
            Processed detections
        """
        # Remove batch dimension if present
        if len(outputs.shape) == 3:
            outputs = outputs[0]
            
        # If no detections, return None
        if outputs.shape[0] == 0:
            return None
        
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
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )[0]
            self.dt = time.time() - t0
            
            # Postprocess
            detections = self.postprocess(outputs)
            results.append(detections)
            
        return results[0] if len(results) == 1 else results 