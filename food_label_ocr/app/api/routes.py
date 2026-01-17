from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/api")

@router.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "status": "received"
    }
