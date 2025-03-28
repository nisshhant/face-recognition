import cv2
import numpy as np
import pickle
import insightface
from insightface.app import FaceAnalysis

# Load InsightFace model (use "buffalo_s" for low specs)
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1)  # Use CPU mode

# Load saved face database
face_db_path = "faces2.pkl"
try:
    with open(face_db_path, "rb") as f:
        known_faces = pickle.load(f)
except (FileNotFoundError, EOFError):  # Handle missing or corrupted file
    known_faces = {}

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)

print("Press 'c' to capture a face for enrollment.")
print("Press 'd' for real-time face detection mode.")
print("Press 'q' to exit.")

mode = None  # Store current mode ('capture' or 'detect')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break  # Exit if the camera is not working

    # Convert frame to RGB (required for InsightFace)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = app.get(frame_rgb)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.normed_embedding  # Get face embedding

        # Capture Mode: Save a new face
        if mode == "capture":
            name = input("Enter Name: ")
            user_id = input("Enter ID: ")
            known_faces[name] = {"id": user_id, "embedding": embedding}

            # Save face data to file
            with open(face_db_path, "wb") as f:
                pickle.dump(known_faces, f)

            print(f"Face saved for {name} (ID: {user_id})")
            mode = None  # Reset mode after capturing

        # Detection Mode: Compare with stored faces
        if mode == "detect":
            found = "Unknown"
            max_similarity = 0  # Track highest similarity

            if known_faces:
                embeddings = np.array([data["embedding"] for data in known_faces.values()])
                similarities = np.dot(embeddings, embedding)

                best_match_idx = np.argmax(similarities)
                max_similarity = similarities[best_match_idx]  # Get highest match score

                if max_similarity > 0.4:  # Threshold
                    found = list(known_faces.keys())[best_match_idx]

            # Convert similarity to percentage
            match_percentage = round(max_similarity * 100, 2)
            label = f"{found} ({match_percentage}%)" if found != "Unknown" else "Unknown"

            # Set bounding box color based on match
            color = (0, 255, 0) if found != "Unknown" else (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit
        break
    elif key == ord("c"):  # Capture Mode
        print("Switched to Face Enrollment Mode. Capture a new face!")
        mode = "capture"
    elif key == ord("d"):  # Detection Mode
        print("Switched to Real-Time Face Detection Mode.")
        mode = "detect"

cap.release()
cv2.destroyAllWindows()
