"""
Flask Web Application for Celebrity Face Recognition

This script initializes the Flask app, handles file uploads, and defines routes for the web interface.
It integrates the face recognition logic (predict.py) and bio generation (bio_generator.py)
to provide a complete user experience.
"""
from flask import Flask, request, jsonify, render_template
import os
import torch
from predict import load_models, get_prediction
from bio_generator import generate_bio
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading models on {device}...")
mtcnn, resnet, clf, le = load_models(device)
print("Models loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = get_prediction(filepath, mtcnn, resnet, clf, le, device)
            
            # Generate bio for the top prediction if available
            if result.get("status") == "success" and result.get("predictions"):
                # Basic logic: Get bio for the most confident face
                # Or we could get it for all. Let's do it for the first one for now or all unique ones.
                # Let's attach bio to each prediction.
                for pred in result["predictions"]:
                    if pred["name"] != "Unknown celebrity":
                        pred["bio"] = generate_bio(pred["name"])
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
