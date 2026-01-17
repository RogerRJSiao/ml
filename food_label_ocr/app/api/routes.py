from fastapi import APIRouter, UploadFile, File
from app.services.file_processing import FileProcessService
from app.services.ocr_processing import OCRProcessService

router = APIRouter(prefix="/api")

@router.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    圖檔OCR上傳+分析端點
    接收圖檔，透過服務層驗證後，儲存到organized檔案夾，並以OCR分析另存到annotated。
    """
    #--處理上傳檔案
    result_upload = await FileProcessService.process_upload(file)
    if result_upload["status"] == "error":
        return result_upload
    
    #--取得已上傳的檔案配置，並進行OCR處理(現為模擬)
    filepath_upload = result_upload["data"].filepath_organized
    result_ocr = await OCRProcessService.process_ocr(filepath_upload)
    return result_ocr
