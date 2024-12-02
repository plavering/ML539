import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    KNOWN_FACES_DIR = DATA_DIR / "known_faces"
    DB_DIR = BASE_DIR / "database" / "face_db"

    # Model configurations
    YOLO_MODEL = "yolov8s.pt"
    FACE_MODEL = "Facenet"
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.7
    IOU_THRESHOLD = 0.7
    POSITION_TIMEOUT = 5.0
    SEARCH_COOLDOWN = 1.0

    # Database settings
    COLLECTION_NAME = "face_embeddings"
    
    # Ensure directories exist
    @classmethod
    def setup_directories(cls):
        for directory in [cls.DATA_DIR, cls.KNOWN_FACES_DIR, cls.DB_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


