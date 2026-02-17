"""
Biography Generator Module

This module interfaces with the Google Gemini API to generate short biographies
for recognized celebrities. It handles API configuration and text generation requests.
"""
import google.generativeai as genai
import os

# Configure the API key
# It's best practice to use environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def generate_bio(celebrity_name):
    """
    Generates a short biography for a celebrity using Google Gemini.
    """
    if not GOOGLE_API_KEY:
        return "Bio generation unavailable: API key not set."

    if celebrity_name.lower() == "unknown celebrity":
        return "No biography available for unknown persons."

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Write a short, engaging biography for the celebrity {celebrity_name}. Keep it under 200 words. Focus on their main achievements."
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Bio generation failed: {e}")
        print("DEBUG: Listing available models...")
        try:
            for m in genai.list_models():
                print(f" - {m.name}")
        except Exception as list_err:
            print(f"Could not list models: {list_err}")
            
        return f"Could not generate bio: {str(e)}"
