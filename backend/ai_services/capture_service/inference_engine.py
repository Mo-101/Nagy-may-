import logging
import cv2
import numpy as np
from typing import Dict, Optional, List
from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Local inference engine using Ultralytics YOLO.
    Eliminates the need for external API calls for detection.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the .pt model file
            conf_threshold: Minimum confidence score for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._load_model()
        
    def _load_model(self):
        """Load the YOLO model within safety blocks"""
        try:
            logger.info(f"[INFERENCE] Loading model from {self.model_path} on {self.device}...")
            self.model = YOLO(self.model_path)
            logger.info(f"[INFERENCE] Model loaded successfully: {self.model.info()}")
        except Exception as e:
            logger.error(f"[INFERENCE] Failed to load model: {e}")
            raise RuntimeError(f"Could not load model at {self.model_path}")

    def predict(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Run inference on a single frame.
        
        Args:
            frame: Numpy array of the image/frame (BGR)
            
        Returns:
            Dictionary containing detection results or None if failed.
            Format matches the previous API response structure for compatibility.
        """
        if self.model is None:
            logger.error("[INFERENCE] Model not initialized")
            return None
            
        try:
            # Run inference
            # stream=True for efficiency if processing video frames sequentially, 
            # but for single frame calls, standard predict is fine.
            results = self.model.predict(
                source=frame, 
                conf=self.conf_threshold, 
                device=self.device, 
                verbose=False
            )
            
            # Process results
            # YOLO results is a list (one per image), we only sent one
            result = results[0]
            
            detections = []
            total_conf = 0
            
            for box in result.boxes:
                # Extract box coordinates (xyxy)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls_id]
                
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class": class_name,
                    "class_id": cls_id
                })
                total_conf += conf
                
            avg_conf = (total_conf / len(detections)) if detections else 0.0
            
            return {
                "detections": detections,
                "count": len(detections),
                "avg_confidence": avg_conf,
                "inference_time": sum(result.speed.values()) # sum of pre, inference, post in ms
            }
            
        except Exception as e:
            logger.error(f"[INFERENCE] Inference error: {e}")
            return None
