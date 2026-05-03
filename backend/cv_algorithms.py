import os
import cv2
import numpy as np

def load_att_dataset(base_path, train_split=6):
    """Loads the AT&T database and splits into train/test sets."""
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    if not os.path.exists(base_path):
        return None, None, None, None
        
    for subject_id in range(1, 41):
        subject_dir = os.path.join(base_path, f's{subject_id}')
        if not os.path.isdir(subject_dir):
            continue
        
        for img_idx in range(1, 11):
            img_path = os.path.join(subject_dir, f'{img_idx}.pgm')
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img_flat = img.flatten()
            
            if img_idx <= train_split:
                X_train.append(img_flat)
                y_train.append(subject_id)
            else:
                X_test.append(img_flat)
                y_test.append(subject_id)
                
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

class EigenFaceRecognizer:
    def __init__(self, num_components=40):
        self.num_components = num_components
        self.mean_face = None
        self.eigenfaces = None
        self.training_weights = None
        self.training_labels = None
        self.face_shape = (112, 92)

    def train(self, X_train, y_train):
        self.training_labels = y_train
        n_samples = X_train.shape[0]
        
        self.mean_face = np.mean(X_train, axis=0)
        Phi = X_train - self.mean_face
        
        # Surrogate covariance matrix L = Phi * Phi^T
        L = np.dot(Phi, Phi.T) / n_samples
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # True eigenfaces
        U = np.dot(Phi.T, eigenvectors[:, :self.num_components]).T
        
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.eigenfaces = U / norms
        self.training_weights = np.dot(Phi, self.eigenfaces.T)

    def predict(self, face_vector):
        Phi_test = face_vector - self.mean_face
        test_weight = np.dot(Phi_test, self.eigenfaces.T)
        
        distances = np.linalg.norm(self.training_weights - test_weight, axis=1)
        best_match_idx = np.argmin(distances)
        return self.training_labels[best_match_idx], float(np.min(distances)), test_weight.tolist()

    def get_distances_for_roc(self, X_test, y_test):
        scores = []
        binary_labels = []
        
        for i, face_vector in enumerate(X_test):
            Phi_test = face_vector - self.mean_face
            test_weight = np.dot(Phi_test, self.eigenfaces.T)
            
            for j, train_weight in enumerate(self.training_weights):
                dist = np.linalg.norm(test_weight - train_weight)
                scores.append(-dist) # negative distance because higher score should mean more likely to be same
                binary_labels.append(1 if y_test[i] == self.training_labels[j] else 0)
                
        return np.array(binary_labels), np.array(scores)

def detect_face(image, template=None):
    """
    Detect face using pre-built library OpenCV Haar Cascades.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Optional: downscale image for faster detection if very large, but Haar cascade does multi-scale anyway.
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return 0, 0, 0, 0
        
    # Get the largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    return int(x), int(y), int(w), int(h)

def calculate_roc(y_true, y_scores):
    """
    ROC and AUC from scratch.
    """
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    
    if tps[-1] == 0 or fps[-1] == 0:
        return [0, 1], [0, 1], 0.5
        
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    
    auc = np.trapz(tpr, fpr)
    return fpr.tolist(), tpr.tolist(), float(auc)
