"""
Model Training Script

This script handles the training process for the face recognition system.
It performs the following steps:
1. Loads images from the dataset directory.
2. Detects faces using MTCNN and generates embeddings using InceptionResnetV1 (FaceNet).
3. Trains a Support Vector Machine (SVM) classifier on the embeddings.
4. Saves the trained model and label encoder for later use.
"""
import os
import glob
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Configuration
DATASET_PATH = r'c:\Antigravity\FaceRecognition\dataset'
MODEL_PATH = r'c:\Antigravity\FaceRecognition\svm_model.pkl'
ENCODER_PATH = r'c:\Antigravity\FaceRecognition\label_encoder.pkl'

def get_image_paths_and_labels(dataset_rect):
    image_paths = []
    labels = []
    
    # Get all subdirectories (celebrity names)
    classes = [d for d in os.listdir(dataset_rect) if os.path.isdir(os.path.join(dataset_rect, d))]
    
    for class_name in classes:
        class_dir = os.path.join(dataset_rect, class_name)
        # Support common image extensions
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in glob.glob(os.path.join(class_dir, ext)):
                image_paths.append(img_path)
                labels.append(class_name)
                
    return image_paths, labels

def generate_embeddings(image_paths, mtcnn, resnet, device):
    embeddings = []
    valid_indices = []
    
    print(f"Generating embeddings for {len(image_paths)} images...")
    
    for i, path in enumerate(image_paths):
        if i % 50 == 0:
            print(f"Processing {i}/{len(image_paths)}")
            
        try:
            img = Image.open(path).convert('RGB')
            # Get cropped face tensor directly
            # mtcnn returns tensor if return_prob=False (default)
            # Returns None if no face detected
            img_cropped = mtcnn(img)
            
            if img_cropped is not None:
                # Calculate embedding (unsqueeze to add batch dimension)
                with torch.no_grad():
                    img_embedding = resnet(img_cropped.unsqueeze(0).to(device))
                
                embeddings.append(img_embedding.cpu().numpy()[0])
                valid_indices.append(i)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
            
    return np.array(embeddings), valid_indices

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize models
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        select_largest=False, keep_all=False, device=device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # 1. Load Data
    print("Loading dataset...")
    image_paths, labels = get_image_paths_and_labels(DATASET_PATH)
    print(f"Found {len(image_paths)} images across {len(set(labels))} classes.")
    
    if len(image_paths) == 0:
        print("No images found! Check dataset path.")
        return

    # 2. Encode Labels
    le = LabelEncoder()
    y_all = le.fit_transform(labels)
    
    # 3. Split Data (Stratified to keep class balance)
    # Note: We split paths first, then generate embeddings only for valid ones
    X_train_paths, X_test_paths, y_train_labels, y_test_labels = train_test_split(
        image_paths, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    
    # 4. Generate Embeddings
    print("\nProcessing Training Set...")
    X_train_emb, train_indices = generate_embeddings(X_train_paths, mtcnn, resnet, device)
    y_train = y_train_labels[train_indices] # Filter labels for detected faces
    
    print(f"\nProcessing Test Set...")
    X_test_emb, test_indices = generate_embeddings(X_test_paths, mtcnn, resnet, device)
    y_test = y_test_labels[test_indices] # Filter labels for detected faces
    
    print(f"\nFinal Training samples: {len(X_train_emb)}")
    print(f"Final Test samples: {len(X_test_emb)}")

    # 5. Train Classifier
    print("\nTraining SVM Classifier...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train_emb, y_train)

    # 6. Evaluate
    print("Evaluating...")
    y_pred = clf.predict(X_test_emb)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # 7. Save Models
    print("Saving models...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved encoder to {ENCODER_PATH}")

if __name__ == '__main__':
    main()
