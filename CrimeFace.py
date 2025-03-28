import cv2
import numpy as np
import pickle
import insightface
from insightface.app import FaceAnalysis
import time

# Load InsightFace model (use "buffalo_s" for low specs)
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1)  # Use CPU mode

# Load saved face database
face_db_path = "faces.pkl"
try:
    with open(face_db_path, "rb") as f:
        known_faces = pickle.load(f)
except (FileNotFoundError, EOFError):  # Handle missing or corrupted file
    known_faces = {}

# Load input image for identity matching
input_image_path = "static/elon.jpg"
input_image = cv2.imread(input_image_path)
if input_image is None:
    print("Error: Could not load input image.")
    exit()

input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
faces = app.get(input_image_rgb)

if not faces:
    print("No faces detected in the input image.")
    exit()

input_embedding = faces[0].normed_embedding  # Use the first detected face

# Identify the closest match in the database
found_name = "Unknown"
found_id = "N/A"
if known_faces:
    embeddings = np.array([data["embedding"] for data in known_faces.values()])
    similarities = np.dot(embeddings, input_embedding)

    best_match_idx = np.argmax(similarities)
    if similarities[best_match_idx] > 0.6:  # Threshold
        found_name = list(known_faces.keys())[best_match_idx]
        found_id = known_faces[found_name]["id"]

print(f"Identified Person: {found_name} (ID: {found_id})")

# Load video for tracking
video_path = "static/elonwa.mp4"  # Change this to your video file path
cap = cv2.VideoCapture(video_path)
appearance_log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to capture frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(frame_rgb)
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get timestamp in seconds

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.normed_embedding

        # Compare with the identified person's embedding
        similarity = np.dot(input_embedding, embedding)
        if similarity > 0.6:  # Threshold for same person
            appearance_log.append(timestamp)
            label = found_name
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Tracking System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit
        break

cap.release()
cv2.destroyAllWindows()

# Save appearance timestamps
log_path = "appearance_log.txt"
with open(log_path, "w") as log_file:
    for ts in appearance_log:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}\n")

print(f"Appearance log saved to {log_path}")