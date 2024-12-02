import multiprocessing as mp
import cv2
import queue
import psutil
import numpy as np
import logging
import time
import os
from main import FaceRecognitionSystem
from deepface import DeepFace
from collections import deque

def suppress_logging():
    for logger in ["ultralytics", "chromadb.telemetry", "main"]:
        logging.getLogger(logger).setLevel(logging.WARNING)

class BoxTracker:
    def __init__(self, history_size=5):
        self.history = {}
        self.history_size = history_size
        self.last_positions = {}
        
    def update(self, detection_id, box):
        if detection_id not in self.history:
            self.history[detection_id] = deque(maxlen=self.history_size)
        
        self.history[detection_id].append(box)
        
    def get_smooth_box(self, detection_id):
        if detection_id not in self.history:
            return None
            
        boxes = np.array(self.history[detection_id])
        if len(boxes) < 2:
            return boxes[-1]
            
        # Apply exponential moving average
        weights = np.exp(np.linspace(-1, 0, len(boxes)))
        weights /= weights.sum()
        smooth_box = np.sum(boxes * weights[:, np.newaxis], axis=0)
        return smooth_box.astype(int)

class FrameProcessor:
    def __init__(self, config_path):
        self.system = FaceRecognitionSystem(config_path)
        self.last_detection_time = {}
        self.detection_cooldown = 0.3  # Reduced from 0.5 for smoother updates
        self.min_confidence = 0.7
        self.box_tracker = BoxTracker(history_size=5)
        self.detection_buffer = {}
        # Add dictionary to track last terminal output for each person
        self.last_terminal_output = {}
        
    def apply_nms(self, boxes, scores, iou_threshold=0.5):
        # Convert boxes to the format expected by NMSBoxes
        boxes_list = [box.tolist() for box in boxes]
        scores_list = [float(score) for score in scores]
        
        indices = cv2.dnn.NMSBoxes(
            boxes_list, 
            scores_list, 
            score_threshold=self.min_confidence, 
            nms_threshold=iou_threshold
        )
        
        # Handle different OpenCV versions' return format
        if isinstance(indices, np.ndarray):
            return indices.flatten()
        return [i[0] for i in indices] if indices else []

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_frame = frame.copy()
        current_time = time.time()
        
        try:
            results = self.system.model.predict(
                source=frame_rgb, 
                show=False, 
                classes=[0],
                verbose=False
            )
            
            # Collect all detections first
            current_detections = []
            for result in results:
                boxes = result.boxes
                for box, conf in zip(boxes.xyxy, boxes.conf):
                    if conf > self.min_confidence:
                        box_np = box.cpu().numpy()
                        current_detections.append((box_np, conf))
            
            # Apply NMS to remove overlapping detections
            if current_detections:
                boxes = [det[0] for det in current_detections]
                scores = [det[1] for det in current_detections]
                indices = self.apply_nms(boxes, scores)
                
                # Process surviving detections
                for idx in indices:
                    box_np = boxes[idx]
                    detection_id = f"{int(box_np[0])}-{int(box_np[1])}"
                    
                    # Update tracker
                    self.box_tracker.update(detection_id, box_np)
                    smooth_box = self.box_tracker.get_smooth_box(detection_id)
                    
                    if smooth_box is not None:
                        if (detection_id not in self.last_detection_time or 
                            current_time - self.last_detection_time[detection_id] >= self.detection_cooldown):
                            
                            self.last_detection_time[detection_id] = current_time
                            
                            x1, y1, x2, y2 = map(int, smooth_box)
                            
                            # Add padding to the detection window
                            pad = 10
                            y1 = max(0, y1 - pad)
                            y2 = min(frame.shape[0], y2 + pad)
                            x1 = max(0, x1 - pad)
                            x2 = min(frame.shape[1], x2 + pad)
                            
                            person_img = frame_rgb[y1:y2, x1:x2]
                            
                            temp_path = f"temp_detection_{current_time}_{detection_id}.jpg"
                            cv2.imwrite(temp_path, cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR))
                            
                            try:
                                embedding = DeepFace.represent(
                                    img_path=temp_path,
                                    model_name="Facenet",
                                    enforce_detection=False
                                )
                                
                                if embedding:
                                    results = self.system.collection.query(
                                        query_embeddings=[embedding[0]['embedding']],
                                        n_results=1
                                    )
                                    
                                    if results['distances'][0]:
                                        match_confidence = 1 - results['distances'][0][0]
                                        person_data = results['metadatas'][0][0]
                                        
                                        if match_confidence > self.min_confidence:
                                            person_name = person_data['name']
                                            # Store detection in buffer
                                            self.detection_buffer[detection_id] = {
                                                'box': (x1, y1, x2, y2),
                                                'confidence': match_confidence,
                                                'name': person_name,
                                                'time': current_time
                                            }
                                            
                                            # Check if we should print to terminal
                                            terminal_cooldown = 10.00  # THIS DOESNT WORK 
                                            if (person_name not in self.last_terminal_output or 
                                                current_time - self.last_terminal_output[person_name] >= terminal_cooldown):
                                                
                                                print(f"\n=== Person Detected ===")
                                                print(f"Name: {person_name}")
                                                print(f"Match Confidence: {match_confidence:.1%}")
                                                if person_data.get('linkedin'):
                                                    print(f"LinkedIn: {person_data['linkedin']}")
                                                print("=====================")
                                                
                                                self.last_terminal_output[person_name] = current_time
                            finally:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
            
            # Render all active detections
            buffer_timeout = 0.5  # Detection stays visible for 0.5 seconds
            for det_id, det_info in list(self.detection_buffer.items()):
                if current_time - det_info['time'] > buffer_timeout:
                    del self.detection_buffer[det_id]
                    continue
                
                x1, y1, x2, y2 = det_info['box']
                match_confidence = det_info['confidence']
                
                # Smooth color transition
                green = int(255 * match_confidence)
                color = (0, green, 0)
                
                # Draw with anti-aliasing
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                
                text = f"{det_info['name']} ({match_confidence:.1%})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                
                # Improved text background
                cv2.rectangle(
                    display_frame,
                    (x1, y1 - text_size[1] - 10),
                    (x1 + text_size[0], y1),
                    color,
                    -1,
                    cv2.LINE_AA
                )
                
                # Anti-aliased text
                cv2.putText(
                    display_frame,
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            
        return display_frame

# Rest of the code remains the same...
def worker_process(frame_queue, result_queue, config_path, worker_id):
    try:
        suppress_logging()
        print(f"Worker {worker_id}: Initializing...")
        
        processor = FrameProcessor(config_path)
        print(f"Worker {worker_id}: Ready")
        
        while True:
            try:
                task = frame_queue.get(timeout=1)
                if task is None:
                    break
                    
                frame_data, frame_id = task
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((480, 640, 3)).copy()
                processed = processor.process_frame(frame)
                result_queue.put((frame_id, processed.tobytes()))
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {str(e)}")
                
    except Exception as e:
        logging.error(f"Worker {worker_id} failed to initialize: {str(e)}")

def display_process(display_queue, quit_event):
    last_fps_update = time.time()
    frames_processed = 0
    fps = 0
    
    while not quit_event.is_set():
        try:
            frame_data = display_queue.get(timeout=0.1)
            if frame_data is None:
                break
                
            frame_id, frame_bytes = frame_data
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((480, 640, 3)).copy()
            
            frames_processed += 1
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                fps = frames_processed / (current_time - last_fps_update)
                frames_processed = 0
                last_fps_update = current_time
            
            fps_text = f"FPS: {fps:.1f}"
            text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            
            cv2.rectangle(
                frame,
                (5, 5),
                (text_size[0] + 15, 40),
                (0, 0, 0),
                -1
            )
            
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            cv2.imshow('Face Recognition (Parallel)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_event.set()
                
        except queue.Empty:
            continue
            
    cv2.destroyAllWindows()

def run_parallel_system(config_path: str, num_processes: int = None):
    if num_processes is None:
        num_processes = max(1, min(4, psutil.cpu_count() - 1))
    
    print(f"\nStarting face recognition with {num_processes} processes...")
    
    frame_queue = mp.Queue(maxsize=num_processes)
    result_queue = mp.Queue(maxsize=num_processes)
    display_queue = mp.Queue(maxsize=2)
    quit_event = mp.Event()
    
    processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=worker_process,
            args=(frame_queue, result_queue, config_path, i)
        )
        p.daemon = True
        p.start()
        processes.append(p)
    
    display_proc = mp.Process(
        target=display_process,
        args=(display_queue, quit_event)
    )
    display_proc.daemon = True
    display_proc.start()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open video capture device")
    
    frame_id = 0
    try:
        print("\nProcessing video feed... (Press 'q' to quit)")
        
        while not quit_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                frame_queue.put((frame.tobytes(), frame_id), block=False)
                frame_id += 1
            except queue.Full:
                continue
            
            try:
                while True:
                    result = result_queue.get_nowait()
                    display_queue.put(result)
            except queue.Empty:
                pass
            
    except KeyboardInterrupt:
        print("\nStopping gracefully...")
    finally:
        print("\nCleaning up...")
        quit_event.set()
        
        for _ in processes:
            frame_queue.put(None)
        display_queue.put(None)
        
        for p in processes:
            p.terminate()
            p.join(timeout=1)
        
        display_proc.terminate()
        display_proc.join(timeout=1)
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("Cleanup complete")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    suppress_logging()
    
    try:
        run_parallel_system("config/people.yaml", num_processes=4)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
