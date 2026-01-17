from fastapi import APIRouter, UploadFile, File
from app.services.file_processing import FileProcessService

router = APIRouter(prefix="/api")

@router.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    圖檔OCR上傳
    接受圖片檔案，透過服務層驗證後，儲存到uploads檔案夾
    """
    result = await FileProcessService.process_upload(file)
    return result
