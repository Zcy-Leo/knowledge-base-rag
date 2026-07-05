import os
import sys
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Marker PDF Parser API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    _marker_available = True
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    print("✅ Marker models loaded successfully")
except Exception as e:
    _marker_available = False
    print(f"⚠️ Marker not available: {e}")


@app.get("/health")
async def health_check():
    return {"status": "ok", "marker_available": _marker_available}


@app.post("/parse/pdf")
async def parse_pdf(file: UploadFile = File(...)):
    if not _marker_available:
        raise HTTPException(status_code=503, detail="Marker not available on this server")
    
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


@app.post("/parse/pdf/batch")
async def parse_pdf_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            rendered = converter(temp_path)
            markdown_text, _, _ = text_from_rendered(rendered)
            
            os.unlink(temp_path)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "markdown_length": len(markdown_text)
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
