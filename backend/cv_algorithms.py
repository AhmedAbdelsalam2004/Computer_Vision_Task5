import os
import cv2
import numpy as np

def load_att_dataset(base_path, train_split=8):
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
    # If the image is exactly or very close to the dataset template size, 
    # assume it's a pre-cropped dataset image and return it whole.
    # This prevents Haar cascade from making a tight crop that ruins PCA alignment.
    if template is not None:
        h_i, w_i = image.shape
        h_t, w_t = template.shape
        if h_i <= h_t + 10 and w_i <= w_t + 10:
            return 0, 0, w_i, h_i
            
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return 0, 0, image.shape[1], image.shape[0]
        
    # Get the largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Expand the bounding box to match the AT&T dataset framing
    # AT&T faces include shoulders and forehead. A Haar cascade is too tight.
    center_x = x + w // 2
    center_y = y + h // 2
    
    new_w = int(w * 1.5)
    new_h = int(new_w * (112 / 92))
    
    # Shift slightly up
    center_y = int(center_y - h * 0.1)
    
    new_x = max(0, center_x - new_w // 2)
    new_y = max(0, center_y - new_h // 2)
    
    new_w = min(new_w, image.shape[1] - new_x)
    new_h = min(new_h, image.shape[0] - new_y)
    
    return int(new_x), int(new_y), int(new_w), int(new_h)

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
