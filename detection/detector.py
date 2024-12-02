from ultralytics import YOLO
import numpy as np
from typing import List, Tuple

class PersonDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_persons(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Detect persons in the given frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of tuples containing bounding boxes and confidence scores
        """
        results = self.model.predict(source=frame, show=False, classes=[0])
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box, conf in zip(boxes.xyxy, boxes.conf):
                if conf > self.confidence_threshold:
                    detections.append((box.cpu().numpy(), conf.item()))
                    
        return detections

