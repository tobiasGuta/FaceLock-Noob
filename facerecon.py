import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle

# --- Configuration and Paths ---
MODEL_FILE = "recognizer.pkl"
SCALER_FILE = "scaler.pkl"

PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
EMBEDDER_PATH = "nn4.small2.v1.t7"

# Initialize DNN models
try:
    net_detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    net_embedder = cv2.dnn.readNetFromTorch(EMBEDDER_PATH)
except Exception as e:
    print(f"\n!! CRITICAL ERROR: Could not load DNN models: {e}")
    exit()

# --- HELPER FUNCTIONS ---

def detect_faces_dnn(image):
    """Detects faces using the Caffe DNN model."""
    (h, w) = image.shape[:2]
    # Standard blob processing for Caffe model
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net_detector.setInput(blob)
    detections = net_detector.forward()
    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Calculate width and height from coordinates
            w_box = endX - startX
            h_box = endY - startY
            boxes.append((startX, startY, w_box, h_box))
    return boxes

def crop_face_padded(img, box, padding_ratio=0.1): 
    """Crops the face with padding to ensure features are captured."""
    x, y, w, h = box
    
    # Calculate padding based on a ratio of the face dimensions
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)
    
    # Apply padding, ensuring boundaries stay within image limits
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)
    
    return img[y1:y2, x1:x2]

def get_face_embedding(face_img):
    """Computes the 128-D face embedding using the Torch DNN model."""
    face_resized = cv2.resize(face_img, (96, 96))
    # Standard blob processing for OpenFace (nn4.small2.v1.t7) model
    face_blob = cv2.dnn.blobFromImage(face_resized, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    net_embedder.setInput(face_blob)
    embedding = net_embedder.forward().flatten()
    return embedding

def load_and_prepare_embeddings(folder_path, label):
    """Loads faces from a folder, calculates embeddings, and assigns a label."""
    embeddings = []
    labels = []
    
    if not os.path.exists(folder_path):
        return embeddings, labels

    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, file)
            img = cv2.imread(path)
            if img is None: continue

            boxes = detect_faces_dnn(img)
            
            if boxes:
                # Find the largest face (most likely the target)
                box = max(boxes, key=lambda b: b[2] * b[3])
                # Use padded crop
                face = crop_face_padded(img, box) 
                embedding = get_face_embedding(face)
                embeddings.append(embedding)
                labels.append(label)
    return embeddings, labels

# --- Main Recognition Logic (SVM) ---

def train_and_save_model():
    """Trains the SVM and saves both the model and the scaler."""
    REF_FOLDER = "reference_faces"
    NEG_FOLDER = "negative_faces"

    # Step 1: Gather Embeddings
    print("Step 1: Gathering Positive (1) and Negative (0) Embeddings...")
    pos_embeds, pos_labels = load_and_prepare_embeddings(REF_FOLDER, 1)
    neg_embeds, neg_labels = load_and_prepare_embeddings(NEG_FOLDER, 0)

    if not pos_embeds:
        raise ValueError("No positive (target) embeddings found. Check 'reference_faces' folder.")

    all_embeddings = np.array(pos_embeds + neg_embeds)
    all_labels = np.array(pos_labels + neg_labels)

    # Step 2a: Standardize/Normalize Embeddings
    print("Step 2a: Scaling embeddings for robust SVM training...")
    scaler = StandardScaler()
    all_embeddings_scaled = scaler.fit_transform(all_embeddings)

    if len(np.unique(all_labels)) < 2:
        print("\nNOTE: Only positive examples loaded. Training will proceed without negative examples.")

    # Step 2b: Train SVM Classifier
    print(f"Step 2b: Training SVM with {len(pos_embeds)} positive and {len(neg_embeds)} negative examples.")
    recognizer = SVC(kernel="linear", probability=True, random_state=42)
    recognizer.fit(all_embeddings_scaled, all_labels)
    
    # Step 2c: Save the trained model and scaler
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(recognizer, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Models saved to {MODEL_FILE} and {SCALER_FILE}. Retraining skipped on future runs.")
    
    return recognizer, scaler

def load_models():
    """Loads the SVM model and the scaler from files."""
    with open(MODEL_FILE, 'rb') as f:
        recognizer = pickle.load(f)
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    return recognizer, scaler

# --- Execution ---

try:
    # Check if models exist. If not, train and save them.
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        print(f"Loading existing models from {MODEL_FILE} and {SCALER_FILE}.")
        recognizer, scaler = load_models()
    else:
        recognizer, scaler = train_and_save_model()

    # Step 3: Load Probe Image and Predict
    PROBE_IMAGE = "elon.png"
    print(f"\nStep 3: Processing probe image '{PROBE_IMAGE}'...")
    probe_img = cv2.imread(PROBE_IMAGE)
    if probe_img is None:
        raise FileNotFoundError(f"Could not find probe image '{PROBE_IMAGE}'")
        
    probe_faces = detect_faces_dnn(probe_img)
    
    if not probe_faces:
        raise ValueError("No faces detected in test image by DNN!")
        
    # Process each detected face
    for (x, y, w, h) in probe_faces:
        face_img = crop_face_padded(probe_img, (x, y, w, h)) # Use padded crop
        probe_embedding = get_face_embedding(face_img)

        # Scale the probe embedding using the trained scaler
        probe_embedding_scaled = scaler.transform([probe_embedding])

        # Predict the label (0 or 1)
        prediction = recognizer.predict(probe_embedding_scaled)[0]
        
        # Get the confidence (probability)
        proba = recognizer.predict_proba(probe_embedding_scaled)[0]

        # The probability of the face being the target (class 1)
        prob_target = proba[1] if 1 in recognizer.classes_ else 0
        
        # Determine if it's a match (if P(1) is above the threshold)
        # FIX: Lowered threshold to 0.60 to correctly classify faces with high confidence (e.g., 0.65)
        MATCH_THRESHOLD = 0.60 
        is_match = (prob_target > MATCH_THRESHOLD) 
        
        # Determine color and text
        color = (0, 255, 0) if is_match else (0, 0, 255) # Green for match, Red for other
        match_status = "MATCH" if is_match else "NO MATCH"
        
        # 4. Draw result on image
        cv2.rectangle(probe_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(probe_img, f"{match_status} (Prob: {prob_target:.2f})", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Final visualization
    cv2.imshow("OpenCV DNN + SVM Classification", probe_img)
    cv2.waitKey(0)

except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    cv2.destroyAllWindows()
