
import cv2
import numpy as np
import pickle
import insightface
from insightface.app import FaceAnalysis
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

# Load InsightFace model
face_model = FaceAnalysis(name="buffalo_s")
face_model.prepare(ctx_id=1)  # Use CPU mode
cap = cv2.VideoCapture(0)

# Load stored faces
face_db_path = "faces.pkl"
try:
    with open(face_db_path, "rb") as f:
        known_faces = pickle.load(f)
except (FileNotFoundError, EOFError):
    known_faces = {}


def offcamera():
    while True: 
        cap.release();
        


# Function to generate video frames
def generate_frames():
    
   
    while True:
        
        success, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_model.get(frame_rgb)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            embedding = face.normed_embedding

            # Compare with stored faces
            found = None
            if known_faces:
                embeddings = np.array([data["embedding"] for data in known_faces.values()])
                similarities = np.dot(embeddings, embedding)
                best_match_idx = np.argmax(similarities)
                if similarities[best_match_idx] > 0.5:  # Threshold
                    found = list(known_faces.keys())[best_match_idx]

            label = found if found else "Unknown"
            color = (0, 255, 0) if found else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    return Response(offcamera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/enroll', methods=['POST'])
def enroll():
    global known_faces

    name = request.form.get("name")
    user_id = request.form.get("user_id")

    success, frame = cap.read()
    if not success:
        return jsonify({"message": "Failed to capture image"}), 400

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_model.get(frame_rgb)

    if not faces:
        return jsonify({"message": "No face detected"}), 400

    face = faces[0]
    embedding = face.normed_embedding
    known_faces[name] = {"id": user_id, "embedding": embedding}

    with open(face_db_path, "wb") as f:
        pickle.dump(known_faces, f)

    return jsonify({"message": f"Face saved for {name} (ID: {user_id})"}), 200

@app.route('/demo')
def demo():
   global stopcame
   stopcame=True;
   return jsonify({"message": "Face saved for"}), 200

@app.route('/start')
def start():
   global stopcame
   stopcame=False;
   video_feed();
   return jsonify({"message": "Start"}), 200

if __name__ == "__main__":
    app.run(debug=True)
