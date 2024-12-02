import cv2
import numpy as np
from typing import Dict, Tuple

def draw_match_overlay(frame: np.ndarray, box: np.ndarray, 
                      match_data: Dict, confidence: float) -> None:
    """Draw recognition results overlay on the frame."""
    if not match_data:
        return

    x1, y1, x2, y2 = map(int, box)
    
    # Define visualization parameters
    bg_color = (0, 0, 0)
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    padding = 5
    line_height = 25

    # Prepare text lines
    lines = [f"Confidence: {confidence:.2f}"]
    
    for i, (metadata, distance) in enumerate(zip(
            match_data['metadatas'][0], match_data['distances'][0])):
        name = metadata.get('name', 'Unknown')
        lines.append(f"Match {i+1}: {name} ({distance:.2f})")

    # Draw overlay
    box_height = len(lines) * line_height + 2 * padding
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1 - box_height), (x2, y1), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw text
    for i, line in enumerate(lines):
        y_pos = y1 - box_height + (i + 1) * line_height
        cv2.putText(frame, line, (x1 + padding, y_pos),
                   font, font_scale, text_color, thickness)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


