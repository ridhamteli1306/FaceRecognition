"""
Prediction Logic and Model Loading

This module contains the core functions for the face recognition system.
It handles:
- Loading the pre-trained FaceNet models and the custom SVM classifier.
- Processing images to detect faces and generate embeddings.
- Making predictions/identifications based on the embeddings.
- Providing results including confidence scores and bounding boxes.
"""
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os

# Configuration
MODEL_PATH = r'c:\Antigravity\FaceRecognition\svm_model.pkl'
ENCODER_PATH = r'c:\Antigravity\FaceRecognition\label_encoder.pkl'

def load_models(device):
    print("Loading FaceNet models...")
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    print("Loading Classifier...")
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
        
    return mtcnn, resnet, clf, le

    return faces

def get_prediction(image_path, mtcnn, resnet, clf, le, device):
    if not os.path.exists(image_path):
        return {"error": "File not found"}

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {"error": f"Error opening image: {e}"}

    # Detect faces
    faces = mtcnn(img)
    
    if faces is None:
        return {"status": "no_faces"}

    # Generate embeddings
    with torch.no_grad():
        embeddings = resnet(faces.to(device)).detach().cpu().numpy()

    # Predict
    predictions = clf.predict(embeddings)
    probs = clf.predict_proba(embeddings)
    
    predicted_names = le.inverse_transform(predictions)
    
    results = []
    
    boxes, _ = mtcnn.detect(img)
    
    for i, (name, prob) in enumerate(zip(predicted_names, probs)):
        confidence = np.max(prob)
        result_entry = {
            "name": name,
            "confidence": float(confidence)
        }
        
        if confidence < 0.5:
            result_entry["name"] = "Unknown celebrity"
            
        if boxes is not None and i < len(boxes):
             result_entry["box"] = boxes[i].tolist()
             
        results.append(result_entry)
            
    return {"status": "success", "predictions": results}

def predict_image(image_path, mtcnn, resnet, clf, le, device):
    # Backward compatibility wrapper for CLI
    result = get_prediction(image_path, mtcnn, resnet, clf, le, device)
    
    if "error" in result:
        print(result["error"])
        return

    if result["status"] == "no_faces":
        print("No faces detected.")
        return

    print(f"Detected {len(result['predictions'])} faces.")
    print("\nResults:")
    print("-" * 30)
    
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for pred in result['predictions']:
        name = pred['name']
        conf = pred['confidence']
        print(f"Found: {name} (Confidence: {conf:.2f})")
        
        if "box" in pred:
            box = pred["box"]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1] - 20), name, fill="red", font=font)

    print("-" * 30)
    
    # Save result
    result_path = "result_" + os.path.basename(image_path)
    save_path = os.path.join(r"C:\Antigravity\FaceRecognition\prediction", result_path)
    img.save(save_path)
    print(f"Result saved to {result_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        return

    image_path = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        mtcnn, resnet, clf, le = load_models(device)
        predict_image(image_path, mtcnn, resnet, clf, le, device)
    except FileNotFoundError:
        print("Models not found. Please run train_model.py first.")

if __name__ == '__main__':
    main()
