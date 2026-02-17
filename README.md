# Celebrity Face Recognition & Biography App

A web-based application that identifies celebrities in uploaded photos and generates short biographies for them using AI.

## Features

-   **Face Detection & Recognition**: Uses FaceNet (InceptionResnetV1) to detect and recognize faces from a pre-trained dataset.
-   **AI Biographies**: Automatically generates a 200-word biography for identified celebrities using Google's Gemini 2.5 Flash model.
-   **Interactive Web UI**: Modern, responsive interface with drag-and-drop upload and real-time results.
-   **Confidence Threshold**: Automatically flags low-confidence predictions (<50%) as "Unknown celebrity".

## Tech Stack

-   **Backend**: Python, Flask
-   **ML/AI**: PyTorch (facenet-pytorch), Scikit-learn (SVM Classifier), Google Gemini API
-   **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

## Setup

### Prerequisites
-   Python 3.8+ installed.
-   A Google Gemini API Key (Get one [here](https://aistudio.google.com/)).

### Installation

1.  Clone or download this repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Ensure you have your trained model files in the root directory:
    -   `svm_model.pkl`
    -   `label_encoder.pkl`
    *(If you haven't trained the model yet, run `train_model.py` first)*

### Configuration

Set your Google API key as an environment variable:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

**Linux/Mac:**
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

## Usage

1.  Start the Flask application:
    ```bash
    python app.py
    ```
2.  Open your web browser and navigate to:
    [http://localhost:5000](http://localhost:5000)
3.  Upload an image of a celebrity to see the prediction and biography.

## Dataset Structure

For the model to learn, organize your images in the `dataset` folder like this:

```text
dataset/
├── Person_Name_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Person_Name_2/
│   ├── photo_A.png
│   ├── photo_B.jpg
│   └── ...
└── ...
```

Each subfolder name will be used as the label (name) for the person.

## Using a Custom Dataset

To train the model on your own set of people:

1.  **Clear existing data**: Delete the default subfolders inside the `dataset/` directory.
2.  **Add your images**:
    -   Create a new folder for each person you want to recognize (e.g., `dataset/Ryan_Reynolds`, `dataset/Scarlett_Johansson`).
    -   Place at least 10-15 clear images of that person inside their folder. Supported formats: `.jpg`, `.jpeg`, `.png`.
3.  **Retrain the model**:
    Run the training script to generate a new `svm_model.pkl` and `label_encoder.pkl`:
    ```bash
    python train_model.py
    ```
4.  **Restart the App**:
    If the web app is running, stop it (Ctrl+C) and start it again to load the new model:
    ```bash
    python app.py
    ```

## Project Structure

-   `app.py`: Flask application entry point.
-   `predict.py`: Core logic for face detection and recognition.
-   `bio_generator.py`: Service to interact with Google Gemini API.
-   `train_model.py`: Script to train the SVM classifier on your dataset.
-   `templates/index.html`: Main web interface.
-   `static/`: CSS and JavaScript files.
-   `setup_dataset.py`: Helper to organize the dataset (if applicable).

## Troubleshooting

-   **404 Model Error**: If you see a Gemini model error, ensure you are using a supported model name in `bio_generator.py`. Currently configured for `gemini-2.5-flash`.
-   **Models not found**: Run `python train_model.py` to regenerate `svm_model.pkl`.

## Customizing the UI

You can modify the look and feel of the website by editing the following files:

-   **HTML Structure**: `templates/index.html` - Change the layout, text, and elements.
-   **Styling (CSS)**: `static/style.css` - Change colors, fonts, animations, and spacing.
-   **bLogic**: `static/script.js` - Change how the frontend handles uploads and displays results.
