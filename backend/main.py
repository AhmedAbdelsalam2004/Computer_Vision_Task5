from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import os
from cv_algorithms import load_att_dataset, EigenFaceRecognizer, detect_face, calculate_roc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer = EigenFaceRecognizer(num_components=40)
dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'att_faces')
X_train, y_train, X_test, y_test = load_att_dataset(dataset_path)

if X_train is not None and len(X_train) > 0:
    recognizer.train(X_train, y_train)

# Cache ROC to avoid heavy recalculation
cached_roc = None

def array_to_base64(arr):
    # Normalize to 0-255
    arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, buffer = cv2.imencode('.png', arr_norm)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/api/status")
def status():
    return {"status": "ok", "trained": recognizer.mean_face is not None}

@app.get("/api/eigenfaces")
def get_eigenfaces():
    if recognizer.eigenfaces is None:
        return {"error": "Model not trained"}
        
    faces_b64 = []
    # Return top 10 eigenfaces
    for i in range(min(10, recognizer.num_components)):
        ef = recognizer.eigenfaces[i].reshape(recognizer.face_shape)
        faces_b64.append(array_to_base64(ef))
        
    mean_face_b64 = array_to_base64(recognizer.mean_face.reshape(recognizer.face_shape))
    
    return {
        "mean_face": mean_face_b64,
        "eigenfaces": faces_b64
    }

@app.get("/api/roc")
def get_roc():
    global cached_roc
    if recognizer.eigenfaces is None or X_test is None:
        return {"error": "Model not trained"}
        
    if cached_roc is not None:
        return cached_roc

    binary_labels, scores = recognizer.get_distances_for_roc(X_test, y_test)
    fpr, tpr, auc = calculate_roc(binary_labels, scores)
    
    # Downsample heavily to 50 points for maximum UI performance
    if len(fpr) > 50:
        indices = np.linspace(0, len(fpr) - 1, 50, dtype=int)
        fpr = [fpr[i] for i in indices]
        tpr = [tpr[i] for i in indices]
    
    cached_roc = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc
    }
    return cached_roc

@app.post("/api/preview")
async def preview(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Invalid image"}
    return {"image_b64": array_to_base64(img)}

@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return {"error": "Invalid image"}
        
    if recognizer.mean_face is None:
        return {"error": "Model not trained on server."}

    # Detect face using NCC
    mean_face_img = recognizer.mean_face.reshape(recognizer.face_shape)
    x, y, w, h = detect_face(img, mean_face_img)
    
    if w == 0 or h == 0:
        return {"error": "No face detected"}
        
    # Crop and resize face
    face_roi = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (recognizer.face_shape[1], recognizer.face_shape[0]))
    face_vector = face_resized.flatten()
    
    # Recognize
    subject_id, distance, _ = recognizer.predict(face_vector)
    
    # Return image with bounding box
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
    _, buffer = cv2.imencode('.png', img_color)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "subject_id": int(subject_id),
        "distance": distance,
        "box": {"x": x, "y": y, "w": w, "h": h},
        "image_b64": img_b64
    }
