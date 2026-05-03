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

def detect_face(image, template):
    """
    Detect face using Normalized Cross-Correlation (NCC) from scratch.
    Uses a memory-efficient strided loop to prevent RAM spikes.
    """
    h_i, w_i = image.shape
    h_t, w_t = template.shape
    
    if h_i < h_t or w_i < w_t:
        return 0, 0, w_i, h_i

    # Downscale for performance if image is very large
    scale = 1.0
    while max(image.shape) > 400:
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
        scale *= 2.0
        
    h_i, w_i = image.shape
    
    # If after scaling it's smaller, adjust template
    if h_i < h_t or w_i < w_t:
        template = cv2.resize(template, (w_i, h_i))
        h_t, w_t = template.shape

    t_mean = np.mean(template)
    t_centered = template - t_mean
    t_norm = np.linalg.norm(t_centered)
    
    if t_norm == 0:
        return 0, 0, w_i, h_i
        
    t_centered /= t_norm

    best_cc = -2
    best_x, best_y = 0, 0
    
    # Stride of 4 pixels to speed up search and save compute
    step = 4
    
    for y in range(0, h_i - h_t + 1, step):
        for x in range(0, w_i - w_t + 1, step):
            window = image[y:y+h_t, x:x+w_t]
            w_mean = np.mean(window)
            w_centered = window - w_mean
            w_norm = np.linalg.norm(w_centered)
            
            if w_norm == 0: 
                w_norm = 1e-10
                
            cc = np.sum(w_centered * t_centered) / w_norm
            if cc > best_cc:
                best_cc = cc
                best_x, best_y = x, y
    
    # Scale back
    x_orig = int(best_x * scale)
    y_orig = int(best_y * scale)
    w_orig = int(w_t * scale)
    h_orig = int(h_t * scale)
    
    return x_orig, y_orig, w_orig, h_orig

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
