"""
Generation Capability Tester

This script iterates through a list of potential Gemini model names (candidates)
and attempts to generate simple content ("Hello") to verify which specific model version
is currently active and working.
"""
import google.generativeai as genai
import os

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = "ENTER_YOUR_API_KEY"
    
genai.configure(api_key=GOOGLE_API_KEY)

candidates = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    "gemini-1.0-pro",
    "gemini-pro"
]

print("Testing candidates...")
found = False
for name in candidates:
    print(f"Testing {name}...", end="")
    try:
        model = genai.GenerativeModel(name)
        response = model.generate_content("Hello")
        print(f" SUCCESS!")
        found = True
        break
    except Exception as e:
        print(f" FAILED.")

if not found:
    print("No working model found.")
