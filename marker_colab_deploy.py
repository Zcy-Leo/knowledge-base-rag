#!/usr/bin/env python
"""
Marker PDF Parser API - Google Colab Deployment Script
Run this in Google Colab with T4 GPU enabled.
"""

# ==============================================
# Step 1: Install Dependencies
# ==============================================
print("="*60)
print("Step 1: Installing dependencies...")
print("="*60)

!pip install marker-pdf[full] fastapi uvicorn python-multipart pyngrok -q

# ==============================================
# Step 2: Download Marker Models
# ==============================================
print("\n" + "="*60)
print("Step 2: Downloading Marker models (first run may take 5-10 minutes)...")
print("="*60)

import marker
marker.download_models()

# ==============================================
# Step 3: Start API Server
# ==============================================
print("\n" + "="*60)
print("Step 3: Starting Marker API Server...")
print("="*60)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import uvicorn
import threading

app = FastAPI(title="Marker PDF Parser API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

model_dict = create_model_dict()
converter = PdfConverter(artifact_dict=model_dict)
print("✅ Marker models loaded successfully")

@app.get("/health")
async def health_check():
    return {"status": "ok", "marker_available": True}

@app.post("/parse/pdf")
async def parse_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name
        
        rendered = converter(temp_path)
        markdown_text, _, _ = text_from_rendered(rendered)
        
        os.unlink(temp_path)
        
        return {
            "success": True,
            "markdown_length": len(markdown_text),
            "content": markdown_text
        }
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

print("\n✅ Marker API Server is running on port 8000")

# ==============================================
# Step 4: Expose to Public via ngrok
# ==============================================
print("\n" + "="*60)
print("Step 4: Exposing server to public...")
print("="*60)

from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

public_url = ngrok.connect(8000)
print(f"\n🚀 Marker API is now available at: {public_url}")
print(f"\nUsage:")
print(f"  marker_api_url = '{public_url}'")
print(f"  kb = extract_knowledge_from_pdf('document.pdf', marker_api_url=marker_api_url)")

# Keep the notebook alive
import time
while True:
    time.sleep(30)
