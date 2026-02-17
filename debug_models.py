"""
Debug Model Lister

This script lists available Gemini models and writes the details (name, supported methods)
to a debug text file 'available_models_debug.txt' for inspection.
"""
import google.generativeai as genai
import os

# Force the key provided earlier if env var is missing
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = "ENTER_YOUR_API_KEY"

genai.configure(api_key=GOOGLE_API_KEY)

print(f"Using key: {GOOGLE_API_KEY[:5]}...")

with open("available_models_debug.txt", "w") as f:
    try:
        f.write("Attempting to list models...\n")
        models = list(genai.list_models())
        f.write(f"Found {len(models)} models.\n")
        for m in models:
            f.write(f"Name: {m.name}\n")
            f.write(f"Supported methods: {m.supported_generation_methods}\n")
            f.write("-" * 20 + "\n")
    except Exception as e:
        f.write(f"Error listing models: {e}\n")
