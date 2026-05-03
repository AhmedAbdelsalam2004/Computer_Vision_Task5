from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2
import base64
import os
from cv_algorithms import load_att_dataset, EigenFaceRecognizer, detect_face, calculate_roc

recognizer = EigenFaceRecognizer(num_components=40)
dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'att_faces')
X_train, y_train, X_test, y_test = load_att_dataset(dataset_path)

if X_train is not None and len(X_train) > 0:
    recognizer.train(X_train, y_train)

cached_roc = None

def array_to_base64(arr):
    arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, buffer = cv2.imencode('.png', arr_norm)
    return base64.b64encode(buffer).decode('utf-8')

def status(request):
    return JsonResponse({"status": "ok", "trained": recognizer.mean_face is not None})

def get_eigenfaces(request):
    if recognizer.eigenfaces is None:
        return JsonResponse({"error": "Model not trained"})
        
    faces_b64 = []
    for i in range(min(10, recognizer.num_components)):
        ef = recognizer.eigenfaces[i].reshape(recognizer.face_shape)
        faces_b64.append(array_to_base64(ef))
        
    mean_face_b64 = array_to_base64(recognizer.mean_face.reshape(recognizer.face_shape))
    
    return JsonResponse({
        "mean_face": mean_face_b64,
        "eigenfaces": faces_b64
    })

def get_roc(request):
    global cached_roc
    if recognizer.eigenfaces is None or X_test is None:
        return JsonResponse({"error": "Model not trained"})
        
    if cached_roc is not None:
        return JsonResponse(cached_roc)

    binary_labels, scores = recognizer.get_distances_for_roc(X_test, y_test)
    fpr, tpr, auc = calculate_roc(binary_labels, scores)
    
    if len(fpr) > 50:
        indices = np.linspace(0, len(fpr) - 1, 50, dtype=int)
        fpr = [fpr[i] for i in indices]
        tpr = [tpr[i] for i in indices]
    
    cached_roc = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc
    }
    return JsonResponse(cached_roc)

@csrf_exempt
def preview(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        contents = file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return JsonResponse({"error": "Invalid image"})
        return JsonResponse({"image_b64": array_to_base64(img)})
    return JsonResponse({"error": "Invalid request"})

@csrf_exempt
def recognize(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        contents = file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return JsonResponse({"error": "Invalid image"})
            
        if recognizer.mean_face is None:
            return JsonResponse({"error": "Model not trained on server."})

        # Detect face
        mean_face_img = recognizer.mean_face.reshape(recognizer.face_shape)
        x, y, w, h = detect_face(img, mean_face_img)
        
        if w == 0 or h == 0:
            return JsonResponse({"error": "No face detected"})
            
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
        
        return JsonResponse({
            "subject_id": int(subject_id),
            "distance": float(distance),
            "box": {"x": x, "y": y, "w": w, "h": h},
            "image_b64": img_b64
        })
    return JsonResponse({"error": "Invalid request"})
