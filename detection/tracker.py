import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Optional
import os

class PersonTracker:
    def __init__(self, config, face_database):
        self.config = config
        self.face_database = face_database
        self.tracked_positions = []
        self.tracked_matches = {}
        self.last_search_time = {}

    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / float(box1_area + box2_area - intersection)

    def process_person(self, frame: np.ndarray, box: np.ndarray, 
                      current_time: float) -> Tuple[int, Optional[Dict]]:
        """
        Process detected person and perform face recognition.
        
        Returns:
            Tuple of (person_index, match_results)
        """
        is_new, person_idx = self.is_new_person(box)
        
        # Extract person image
        x1, y1, x2, y2 = map(int, box)
        person_img = frame[y1:y2, x1:x2]
        
        # Save image temporarily
        temp_path = f"temp_person_{person_idx}.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR))

        matches = None
        
        # Check search cooldown
        if (person_idx not in self.last_search_time or 
            current_time - self.last_search_time.get(person_idx, 0) >= self.config.SEARCH_COOLDOWN):
            
            matches = self.face_database.search_face(temp_path)
            self.last_search_time[person_idx] = current_time
            
            if matches:
                self.tracked_matches[person_idx] = matches

        os.remove(temp_path)
        return person_idx, self.tracked_matches.get(person_idx)

    def is_new_person(self, box: np.ndarray) -> Tuple[bool, int]:
        """Determine if detected person is new or already being tracked."""
        current_time = time.time()
        self.tracked_positions = [(pos, t) for pos, t in self.tracked_positions 
                                if current_time - t < self.config.POSITION_TIMEOUT]
        
        for i, (tracked_box, _) in enumerate(self.tracked_positions):
            if self.calculate_iou(box, tracked_box) > self.config.IOU_THRESHOLD:
                return False, i

        new_index = len(self.tracked_positions)
        self.tracked_positions.append((box, current_time))
        return True, new_index

