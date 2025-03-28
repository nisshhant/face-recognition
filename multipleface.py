import cv2
import numpy as np
import pickle
import os
import insightface
from insightface.app import FaceAnalysis
from tqdm import tqdm

# Path to dataset and database file
DATASET_PATH = "dataset"  # Replace with the actual path
FACE_DB_PATH = "celeb.pkl"

# Initialize InsightFace model
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=-1)  # Use CPU mode

known_faces = {}

# Iterate through each folder (person)
for person in tqdm(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)
        if faces:
            embeddings.append(faces[0].normed_embedding)
    
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        known_faces[person] = {"embedding": avg_embedding}

# Save the database
with open(FACE_DB_PATH, "wb") as f:
    pickle.dump(known_faces, f)

print(f"Database created with {len(known_faces)} individuals.")
