"""
Model Availability Checker V2

An improved utility script to list available Google Gemini models.
Includes library version checks, robust error handling, and verifies
support for 'generateContent'.
"""
import google.generativeai as genai
import os
import sys

print(f"Python version: {sys.version}")
try:
    import importlib.metadata
    print("importlib.metadata imported")
except ImportError:
    print("importlib.metadata failed to import")

try:
    print(f"genai version: {genai.__version__}")
except:
    print("Could not get genai version")

# Fallback API Key check
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = "ENTER_YOUR_API_KEY"

genai.configure(api_key=GOOGLE_API_KEY)

print("Listing supported models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
