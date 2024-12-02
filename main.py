import cv2
import yaml
import logging
import time
import os
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
import chromadb
from chromadb.config import Settings
import numpy as np
from collections import deque

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress YOLO and other verbose logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)

class PersonTracker:
    def __init__(self, max_history=10):
        self.tracked_people = {}  # bbox -> person_id mapping
        self.last_seen = {}  # person_id -> timestamp mapping
        self.confidence_history = {}  # person_id -> deque of recent confidences
        self.max_history = max_history
        self.detection_cooldown = 3.0  # seconds
        self.iou_threshold = 0.9
        self.confidence_threshold = 0.70  # minimum confidence to consider a match valid

    def calculate_iou(self, box1, box2):
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

    def update_person(self, bbox, person_id, confidence):
        """Update tracking information for a person."""
        current_time = time.time()
        
        if person_id not in self.confidence_history:
            self.confidence_history[person_id] = deque(maxlen=self.max_history)
        
        self.confidence_history[person_id].append(confidence)
        self.tracked_people[tuple(bbox)] = person_id
        self.last_seen[person_id] = current_time

    def get_tracked_person(self, bbox):
        """Get the person ID for a given bbox if it matches a tracked person."""
        current_time = time.time()
        
        # Remove old tracking info
        self.tracked_people = {
            box: pid for box, pid in self.tracked_people.items()
            if current_time - self.last_seen[pid] < self.detection_cooldown
        }
        
        # Check for matching bbox
        for tracked_box, person_id in self.tracked_people.items():
            if self.calculate_iou(bbox, tracked_box) > self.iou_threshold:
                if current_time - self.last_seen[person_id] < self.detection_cooldown:
                    return person_id
        
        return None

    def get_average_confidence(self, person_id):
        """Get the average confidence for a person over recent history."""
        if person_id in self.confidence_history:
            confidences = self.confidence_history[person_id]
            if confidences:
                return sum(confidences) / len(confidences)
        return 0.0

class FaceRecognitionSystem:
    def __init__(self, config_path: str):
        self.people_data = self.load_config(config_path)
        self.init_database()
        self.model = YOLO("yolov8s.pt")
        self.person_tracker = PersonTracker()
        
    def load_config(self, config_path: str) -> dict:
        """Load people configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration with {len(config['people'])} people")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def init_database(self):
        """Initialize the face database and populate it with configured faces."""
        try:
            self.db_client = chromadb.Client(Settings(
                persist_directory="face_db",
                is_persistent=False
            ))
            
            self.collection = self.db_client.get_or_create_collection(
                name="face_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            
            for person in self.people_data['people']:
                self.add_person_to_database(person)
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def add_person_to_database(self, person: dict):
        """Add a person to the face database."""
        try:
            if person.get('image_path'):  # Only add if image path is provided
                embedding = DeepFace.represent(
                    img_path=person['image_path'], 
                    model_name="Facenet", 
                    enforce_detection=True
                )
                
                if embedding:
                    person_id = str(abs(hash(person['name'])))
                    self.collection.add(
                        embeddings=[embedding[0]['embedding']],
                        ids=[person_id],
                        metadatas=[person]
                    )
                    logger.info(f"Added {person['name']} to database")
                    
        except Exception as e:
            logger.error(f"Error adding {person['name']}: {str(e)}")

    def process_frame(self, frame):
        """Process a single frame and return annotated frame."""
        try:
            results = self.model.predict(source=frame, show=False, classes=[0], verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box, conf in zip(boxes.xyxy, boxes.conf):
                    if conf > 0.4:  # Person detection confidence
                        self.process_detection(frame, box.cpu().numpy(), conf)
                        
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame

    def process_detection(self, frame, box, confidence):
        """Process a single person detection."""
        tracked_person_id = self.person_tracker.get_tracked_person(box)
        
        if tracked_person_id:
            # Update display for tracked person
            avg_confidence = self.person_tracker.get_average_confidence(tracked_person_id)
            if avg_confidence > self.person_tracker.confidence_threshold:
                person_data = self.get_person_data(tracked_person_id)
                if person_data:
                    self.draw_detection(frame, box, person_data, avg_confidence)
            return
            
        x1, y1, x2, y2 = map(int, box)
        person_img = frame[y1:y2, x1:x2]
        
        try:
            temp_path = f"temp_detection_{time.time()}.jpg"
            cv2.imwrite(temp_path, person_img)
            
            embedding = DeepFace.represent(
                img_path=temp_path, 
                model_name="Facenet", 
                enforce_detection=False
            )
            
            if embedding:
                results = self.collection.query(
                    query_embeddings=[embedding[0]['embedding']],
                    n_results=1
                )
                
                if results['distances'][0]:
                    match_distance = results['distances'][0][0]
                    match_confidence = 1 - match_distance
                    person_data = results['metadatas'][0][0]
                    person_id = str(abs(hash(person_data['name'])))
                    
                    if match_confidence > self.person_tracker.confidence_threshold:
                        self.person_tracker.update_person(box, person_id, match_confidence)
                        
                        # Only print if significant confidence change
                        last_conf = self.person_tracker.get_average_confidence(person_id)
                        if abs(match_confidence - last_conf) > 0.1:  # 10% confidence change
                            print("\n=== Person Detected ===")
                            print(f"Name: {person_data['name']}")
                            print(f"Match Confidence: {match_confidence:.1%}")
                            if person_data.get('linkedin'):
                                print(f"LinkedIn: {person_data['linkedin']}")
                            print("=====================")
                        
                        self.draw_detection(frame, box, person_data, match_confidence)
            
            os.remove(temp_path)
            
        except Exception as e:
            logger.debug(f"No face detected in person detection: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_person_data(self, person_id):
        """Get person data from database by ID."""
        try:
            results = self.collection.get(ids=[person_id])
            if results and results['metadatas']:
                return results['metadatas'][0]
        except Exception as e:
            logger.error(f"Error getting person data: {str(e)}")
        return None

    def draw_detection(self, frame, box, person_data, confidence):
        """Draw detection box and information on frame."""
        x1, y1, x2, y2 = map(int, box)
        
        color = (0, int(255 * confidence), 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        text = f"{person_data['name']} ({confidence:.1%})"
        cv2.putText(frame, text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def run(self):
        """Run the face recognition system on webcam feed."""
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frame = self.process_frame(frame)
                cv2.imshow('Face Recognition', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceRecognitionSystem("config/people.yaml")
    system.run()
