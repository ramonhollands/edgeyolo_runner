from typing import Tuple, Optional
from .onnx_detector import ONNXDetector
from .tflite_detector import TFLiteDetector
from .coreml_detector import CoreMLDetector


class DetectorFactory:
    """Factory class to create detectors based on model type."""
    
    @staticmethod
    def create_detector(
        model_path: str,
        detector_type: str = "auto",
        conf_thres: float = 0.25,
        nms_thres: float = 0.5,
        fp16: bool = False,
        use_acceleration: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """Create a detector based on model type.
        
        Args:
            model_path: Path to the model file
            detector_type: Type of detector ("onnx", "tflite", "coreml", or "auto" for auto-detection)
            conf_thres: Confidence threshold for detections
            nms_thres: Non-maximum suppression threshold
            fp16: Whether to use FP16 precision
            use_acceleration: Whether to use hardware acceleration (CUDA for ONNX, GPU for TFLite, Neural Engine/GPU for CoreML)
            input_size: Model input size (height, width). If None, will be extracted automatically from the model.
            
        Returns:
            Detector instance
        """
        if detector_type == "auto":
            if model_path.lower().endswith('.onnx'):
                detector_type = "onnx"
            elif model_path.lower().endswith('.tflite'):
                detector_type = "tflite"
            elif model_path.lower().endswith(('.mlmodel', '.mlpackage')):
                detector_type = "coreml"
            else:
                raise ValueError(f"Cannot auto-detect model type from path: {model_path}. "
                               "Please specify detector_type explicitly.")
        
        if detector_type == "onnx":
            return ONNXDetector(
                model_path=model_path,
                conf_thres=conf_thres,
                nms_thres=nms_thres,
                fp16=fp16,
                use_cuda=use_acceleration,
                input_size=input_size,
            )
        elif detector_type == "tflite":
            return TFLiteDetector(
                model_path=model_path,
                conf_thres=conf_thres,
                nms_thres=nms_thres,
                fp16=fp16,
                use_gpu=use_acceleration,
                input_size=input_size,
            )
        elif detector_type == "coreml":
            return CoreMLDetector(
                model_path=model_path,
                conf_thres=conf_thres,
                nms_thres=nms_thres,
                fp16=fp16,
                use_acceleration=use_acceleration,
                input_size=input_size,
            )
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}. "
                           "Supported types are: 'onnx', 'tflite', 'coreml'") 