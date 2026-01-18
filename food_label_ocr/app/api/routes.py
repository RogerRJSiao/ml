from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
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
    filepath_upload = result_upload["data"]["filepath_organized"]
    result_ocr = await OCRProcessService.process_ocr(filepath_upload)
    return result_ocr

@router.get("/download/annotated/{filename}")
async def download_annotated(filename: str):
    """
    下載標註後的圖片(如GET /api/download/annotated/tag_20260117_215633_8710.jpg)
    Args:       filename (str): 標註圖檔名稱 (例如: tag_20260117_215633_8710.jpg)      
    """
    result_download = await FileProcessService.download_annotated_image(filename)
    
    #--檢查是否成功下載
    if result_download["status"] == "error":
        return result_download
    return FileResponse(
        path=result_download["data"]["filepath_annotated"],
        media_type="image/jpeg",
        filename=filename
    )
    