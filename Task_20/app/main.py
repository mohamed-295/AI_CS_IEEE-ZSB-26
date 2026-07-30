from fastapi import FastAPI, File, UploadFile
from app.model import OCRModel

app = FastAPI(title="Arabic OCR API")

ocr_system = OCRModel()

@app.get("/")
async def health_check():
    """Health check endpoint returning system status."""
    return {"status": "healthy", "model_loaded": ocr_system.model is not None}

@app.post("/predict")
async def predict_text(file: UploadFile = File(...)):
    """Receives an image file, runs inference, and returns recognized Arabic text."""
    image_bytes = await file.read()
    
    predicted_text = ocr_system.predict(image_bytes)
    
    # Return JSON structure required 
    return {
        "filename": file.filename,
        "predicted_text": predicted_text
    }