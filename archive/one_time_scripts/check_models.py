#!/usr/bin/env python3
"""
Quick script to check available Gemini models
"""
import os
import google.generativeai as genai

# Set up API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print(" GEMINI_API_KEY not set")
    exit(1)

genai.configure(api_key=api_key)

print(" Available Gemini models:")
try:
    models = genai.list_models()
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            print(f" {model.name}")
except Exception as e:
    print(f" Error listing models: {e}")