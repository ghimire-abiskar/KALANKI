import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import collections
import uuid
from scipy.spatial.distance import cosine
import torch
from torchvision import models, transforms
import os

# Load YOLOv8 model for person detection
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=50)  # Tracks disappear after 50 frames

# Confidence threshold for YOLO detections
CONFIDENCE_THRESHOLD = 0.4  

# Dictionary to map DeepSORT IDs to unique global IDs
global_id_map = {}

# Store appearance embeddings for re-identification
appearance_db = {}

# Track bounding box history for smoothing
track_history = collections.defaultdict(list)

# Load feature extraction model
feature_model = models.resnet18(pretrained=True)
feature_model = torch.nn.Sequential(*(list(feature_model.children())[:-1]))
feature_model.eval()

# Transformation function for input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped_person = image[y1:y2, x1:x2]
    if cropped_person.size == 0:
        return None
    
    person_tensor = transform(cropped_person).unsqueeze(0)
    with torch.no_grad():
        features = feature_model(person_tensor)
    return features.squeeze().numpy()

def find_best_match(new_embedding):
    best_match, best_score = None, 0.5
    for old_id, old_embedding in appearance_db.items():
        similarity = 1 - cosine(new_embedding, old_embedding)
        if similarity > best_score:
            best_match, best_score = old_id, similarity
    return best_match

def run_tracking(input_video_path, output_video_path):
    """
    Processes the input video, performs ID tracking, and saves the output.
    This function is called from views.py.
    """

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return False

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        results = model(frame)

        for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
            if int(cls) == 0 and conf.item() > CONFIDENCE_THRESHOLD:
                detections.append([box.tolist(), conf.item()])

        # Track objects with DeepSORT
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, w, h = map(int, track.to_tlwh())
            bbox = (x1, y1, x1 + w, y1 + h)

            # Extract appearance features
            new_features = extract_features(frame, bbox)
            if new_features is None:
                continue

            # ID assignment
            if track_id not in global_id_map:
                best_match = find_best_match(new_features)
                if best_match:
                    global_id_map[track_id] = best_match
                else:
                    new_unique_id = str(uuid.uuid4())[:8]
                    global_id_map[track_id] = new_unique_id
                    appearance_db[new_unique_id] = new_features

            unique_id = global_id_map[track_id]

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {unique_id}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processing complete! Output saved to {output_video_path}")
    return True
