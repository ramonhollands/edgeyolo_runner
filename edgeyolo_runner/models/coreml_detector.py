import numpy as np
import cv2
from typing import Any, List, Union, Tuple, Optional
import time
from PIL import Image

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


class CoreMLDetector:
    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.25,
        nms_thres: float = 0.5,
        fp16: bool = False,
        use_acceleration: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """Initialize the CoreML detector.
        
        Args:
            model_path: Path to the CoreML model file (.mlmodel or .mlpackage)
            conf_thres: Confidence threshold for detections
            nms_thres: Non-maximum suppression threshold
            fp16: Whether to use FP16 precision
            use_acceleration: Whether to use hardware acceleration (Neural Engine, GPU)
            input_size: Model input size (height, width). If None, will be extracted from model.
        """
        if not COREML_AVAILABLE:
            raise ImportError("CoreML Tools is not installed. Please install it to use CoreMLDetector: pip install coremltools")
        
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.fp16 = fp16
        self.is_rotated = False

        # Load CoreML model
        try:
            if use_acceleration:
                # Try to use Neural Engine first, then GPU, then CPU
                compute_units = ct.ComputeUnit.ALL
            else:
                compute_units = ct.ComputeUnit.CPU_ONLY
                
            self.model = ct.models.MLModel(model_path, compute_units=compute_units)
            print(f"CoreML Model loaded: {model_path}")
            print(f"Using compute units: {compute_units}")
            
        except Exception as e:
            print(f"Error loading CoreML model: {e}")
            # Fallback to CPU only
            self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
            print("Falling back to CPU_ONLY compute units")
        
        # Get model info
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.output_name = spec.description.output[0].name
        
        # Check input type
        self.input_desc = spec.description.input[0]
        self.is_image_input = self.input_desc.type.HasField('imageType')
        
        # Extract input size from model if not provided
        if input_size is None:
            self.input_size = self._extract_input_size()
        else:
            self.input_size = input_size
        
        print(f"Input name: {self.input_name}")
        print(f"Output name: {self.output_name}")
        print(f"Input size: {self.input_size}")
        print(f"Input type: {'Image' if self.is_image_input else 'MultiArray'}")
        print(f"Using FP16: {fp16}")

    def _extract_input_size(self) -> Tuple[int, int]:
        """Extract input size from CoreML model.
        
        Returns:
            Tuple of (height, width)
        """
        spec = self.model.get_spec()
        input_desc = spec.description.input[0]
        
        if input_desc.type.HasField('imageType'):
            # Image input type
            if input_desc.type.imageType.HasField('imageSizeRange'):
                # Variable size image
                size_range = input_desc.type.imageType.imageSizeRange
                # Use the maximum size if available, or fallback to default
                height = size_range.heightRange.upperBound if size_range.heightRange.upperBound > 0 else 640
                width = size_range.widthRange.upperBound if size_range.widthRange.upperBound > 0 else 640
            else:
                # Fixed size image
                height = input_desc.type.imageType.height
                width = input_desc.type.imageType.width
        elif input_desc.type.HasField('multiArrayType'):
            # Multi-array input type
            shape = input_desc.type.multiArrayType.shape
            if len(shape) == 4:
                # Assume shape is [batch, channels, height, width] or [batch, height, width, channels]
                if shape[1] == 3 or shape[1] == 1:  # NCHW format
                    height, width = shape[2], shape[3]
                else:  # NHWC format
                    height, width = shape[1], shape[2]
            elif len(shape) == 3:
                # Assume shape is [channels, height, width] or [height, width, channels]
                if shape[0] == 3 or shape[0] == 1:  # CHW format
                    height, width = shape[1], shape[2]
                else:  # HWC format
                    height, width = shape[0], shape[1]
            else:
                raise ValueError(f"Unexpected input shape: {shape}")
        else:
            raise ValueError("Unsupported input type for automatic size extraction")
        
        # Handle dynamic dimensions
        if height <= 0:
            height = 640  # default fallback
        if width <= 0:
            width = 640  # default fallback
            
        return (height, width)

    def preprocess(self, img: np.ndarray) -> Union[np.ndarray, Any]:
        """Preprocess image for inference.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Preprocessed image (PIL Image for image input, numpy array for multiarray input)
        """
        height, width = self.input_size

        # print(f"Input shape: {img.shape}")
        
        # rotate image 90 degrees if needed
        if img.shape[0] > img.shape[1]:
            self.is_rotated = True
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            self.is_rotated = False

        # print("is rotated: ", self.is_rotated)
            
        h, w = img.shape[:2]
        self.ratio_h = height / h
        self.ratio_w = width / w
        new_h, new_w = int(h * self.ratio_h), int(w * self.ratio_w)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create canvas and paste image
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized
        
        # Convert BGR to RGB for CoreML
        # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        return Image.fromarray(canvas)

    def postprocess(self, outputs) -> Optional[np.ndarray]:
        """Postprocess raw model outputs.
        
        Args:
            outputs: Raw model outputs (can be dict or numpy array)
            
        Returns:
            Processed detections
        """
        # Handle dictionary output from CoreML
        if isinstance(outputs, dict):
            if 'output' in outputs:
                outputs = outputs['output']
            else:
                # Use the first output if 'output' key doesn't exist
                outputs = list(outputs.values())[0]
        
        # Remove batch dimension if present
        if len(outputs.shape) == 3:
            outputs = outputs[0]
            
        # If no detections, return None
        if outputs.shape[0] == 0:
            return None
        
        valid_detections = []
        
        # Process detections up to 525 (or available detections)
        num_detections = min(525, outputs.shape[0])
        
        for i in range(num_detections):
            detection = outputs[i]
            
            # Extract confidence (index 4)
            conf = detection[4]
            
            # Filter by confidence threshold
            if conf > self.conf_thres:
                
                # Extract position (indices 0, 1)
                x, y = detection[:2]
                
                # Extract stride (index 6)
                stride = detection[6]
                
                # Extract log width and height (indices 2, 3) and apply exponential with stride
                log_w, log_h = detection[2], detection[3]
                width = np.exp(log_w) * stride
                height = np.exp(log_h) * stride
                
                # Extract class ID (assuming it's at index 5, adjust if needed)
                class_id = detection[5] if len(detection) > 5 else 0
                
                # Convert to standard format: [x, y, w, h, conf, class_id]
                processed_detection = [x, y, width, height, conf, class_id]
                valid_detections.append(processed_detection)
        
        if not valid_detections:
            return None
        
        # Convert to numpy array
        detections = np.array(valid_detections)
        boxes = detections[:, :4]
        scores = detections[:, 4]
        class_ids = detections[:, 5]
        
        # Handle rotation if needed
        if self.is_rotated:
            xs = boxes[:, 0].copy()
            ys = boxes[:, 1].copy()
            widths = boxes[:, 2].copy()
            heights = boxes[:, 3].copy()

            boxes[:, 0] = ys
            boxes[:, 1] = self.input_size[1] - xs
            boxes[:, 2] = heights
            boxes[:, 3] = widths

        # Scale boxes back to original image size
        boxes[:, 0] /= self.ratio_h
        boxes[:, 1] /= self.ratio_w
        boxes[:, 2] /= self.ratio_h
        boxes[:, 3] /= self.ratio_w
        
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
            prediction = self.model.predict({self.input_name: input_tensor})
            self.dt = time.time() - t0
            
            # Postprocess (pass the raw prediction to handle dict/array formats)
            detections = self.postprocess(prediction)
            results.append(detections)
            
        return results[0] if len(results) == 1 else results 