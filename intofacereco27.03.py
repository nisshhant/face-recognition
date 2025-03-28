import cv2
import numpy as np
import pickle
import insightface
from insightface.app import FaceAnalysis

# Load InsightFace model (use "buffalo_s" for low specs)
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1)  # CPU mode

# Load saved face database
try:
    with open("faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
except FileNotFoundError:
    known_faces = {}

# Load video file (fix path)
video_path = r"C:\Users\hp\Downloads\sadface.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    # Convert frame to RGB (required for InsightFace)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = app.get(frame_rgb)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.normed_embedding  # Get face embedding

        # Compare with stored faces
        found = None
        for name, data in known_faces.items():
            similarity = np.dot(embedding, data["embedding"])
            if similarity > 0.6:  # Threshold
                found = name
                break

        label = f"{found}" if found else "Unknown"
        color = (0, 255, 0) if found else (0, 0, 255)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save new face
        if found is None:
            name = input("Enter Name: ")
            user_id = input("Enter ID: ")
            known_faces[name] = {"id": user_id, "embedding": embedding}
            with open("faces.pkl", "wb") as f:
                pickle.dump(known_faces, f)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


