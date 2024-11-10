from ultralytics import YOLO
import numpy as np
import cv2
import time
import os
from datetime import datetime
from deepface import DeepFace
from mss import mss

# Initialize
model = YOLO("yolov8s.pt")

save_dir = "person_detections"
os.makedirs(save_dir, exist_ok=True)

# Choose input source
USE_WEBCAM = True  # Set to False to use screen capture instead

sct = mss()


class PersonTracker:
    def __init__(self):
        self.tracked_positions = []
        self.tracked_analysis = {}  # Store analysis results for each tracked person
        self.iou_threshold = 0.5
        self.position_timeout = 5

    def calculate_iou(self, box1, box2):
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

    def is_new_person(self, box):
        current_time = time.time()

        # Update tracked positions and remove old ones
        new_tracked_positions = []
        for i, (pos, t) in enumerate(self.tracked_positions):
            if current_time - t < self.position_timeout:
                new_tracked_positions.append((pos, t))
            else:
                # Remove analysis for expired positions
                if i in self.tracked_analysis:
                    del self.tracked_analysis[i]

        self.tracked_positions = new_tracked_positions

        # Check if this position overlaps with any tracked position
        for i, (tracked_box, _) in enumerate(self.tracked_positions):
            if self.calculate_iou(box, tracked_box) > self.iou_threshold:
                return False, i

        # If we get here, this is a new person
        new_index = len(self.tracked_positions)
        self.tracked_positions.append((box, current_time))
        return True, new_index

    def update_analysis(self, index, analysis):
        self.tracked_analysis[index] = analysis

    def get_analysis(self, index):
        return self.tracked_analysis.get(index)


def capture_screen():
    screen = np.array(ImageGrab.grab())
    return cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)


def analyze_person(person_img):
    try:
        results = DeepFace.analyze(person_img, actions=['age', 'gender', 'race'], enforce_detection=False)
        if isinstance(results, list) and len(results) > 0:
            return results[0]
        return results
    except Exception as e:
        print(f"Could not analyze face: {str(e)}")
        return None


def draw_analysis_overlay(frame, box, analysis):
    if not analysis:
        return

    x1, y1, x2, y2 = map(int, box)

    # Define colors and font
    bg_color = (0, 0, 0)
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Prepare analysis text
    age = analysis.get('age', 'unknown')
    gender = analysis.get('dominant_gender', 'unknown')
    race = analysis.get('dominant_race', 'unknown')

    # Create text lines
    lines = [
        f"Age: {age}",
        f"Gender: {gender}",
        f"Race: {race}"
    ]

    # Calculate text sizes and positions
    padding = 5
    line_height = 25
    box_height = len(lines) * line_height + 2 * padding

    # Draw semi-transparent background
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


def save_and_analyze_person(frame, box, conf):
    x1, y1, x2, y2 = map(int, box)
    person_img = frame[y1:y2, x1:x2]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{save_dir}/person_{timestamp}_conf_{conf:.2f}.jpg"

    cv2.imwrite(filename, cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR))
    result = analyze_person(person_img)

    return filename, result


def capture_screen():
    try:
        # Capture the main monitor
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)

        # Convert to numpy array
        frame = np.array(screenshot)

        # Convert from BGRA to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        return frame
    except Exception as e:
        print(f"Error capturing screen: {str(e)}")
        return None


def main():
    tracker = PersonTracker()
    person_count = 0
    cap = None

    try:
        if USE_WEBCAM:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return
            print("Using webcam input")
        else:
            print("Using screen capture input")

        while True:
            if USE_WEBCAM:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame from webcam")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = capture_screen()
                if frame is None:
                    print("Error capturing screen")
                    break

            # Create a copy for display
            display_frame = frame.copy()

            # Run YOLO detection
            results = model.predict(source=frame, show=False, classes=[0])  # class 0 is person

            # Process detections
            for result in results:
                boxes = result.boxes
                for box, conf in zip(boxes.xyxy, boxes.conf):
                    if conf > 0.5:  # Confidence threshold
                        box_np = box.cpu().numpy()

                        # Check if this is a new person
                        is_new, person_idx = tracker.is_new_person(box_np)

                        if is_new:
                            filename, analysis = save_and_analyze_person(frame, box_np, conf)
                            person_count += 1
                            if analysis:
                                tracker.update_analysis(person_idx, analysis)
                                print(f"New person detected #{person_count}")

                        # Draw overlay with analysis (either new or existing)
                        analysis = tracker.get_analysis(person_idx)
                        draw_analysis_overlay(display_frame, box_np, analysis)

            # Scale down the display frame for better performance
            scale_percent = 50  # percent of original size
            width = int(display_frame.shape[1] * scale_percent / 100)
            height = int(display_frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            display_frame_resized = cv2.resize(display_frame, dim, interpolation=cv2.INTER_AREA)

            # Show the frame
            cv2.imshow('Detection', cv2.cvtColor(display_frame_resized, cv2.COLOR_RGB2BGR))

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Detection stopped by user")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"Total unique persons detected and saved: {person_count}")


if __name__ == "__main__":
    main()