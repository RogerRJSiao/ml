"""
檔案處理服務
負責上下載檔案的業務邏輯處理
"""
from fastapi import UploadFile, HTTPException
from app.utils.file_utils import FileConfig, FileValidator

class FileProcessService:
    """檔案上下載服務類別"""
    
    @staticmethod
    async def process_upload(file: UploadFile) -> dict:
        """
        處理檔案上傳的完整流程
        Args:       file (UploadFile) 上載檔案
        Returns:    res (dict) 上載狀態、檔名、完整路徑、訊息
        # Raises:     HTTPException 當檔案驗證失敗時拋出
        """
        #--初始化回傳結果
        res = {}
        
        #--驗證副檔名是否允許
        is_valid, msg = FileValidator.validate_extension(file.filename)
        if not is_valid:
            # raise HTTPException(status_code=400, detail=msg)
            res = {
                "status": "error",
                "data": "",
                "msg": msg
            }
            return res
        
        #--讀取檔案內容，非同步操作
        #--FastAPI原生支持async，建立非同步端點
        #--避免阻塞只為等待伺服器讀取此檔案，允許請求也可執行
        #--只能在async def函數內使用await，await到讀取完成
        content = await file.read()

        #--確保整理(上載)、標註(下載)目錄存在
        FileConfig.get_organized_dir()
        # FileConfig.get_annotated_dir()

        #--建立整理用的檔案路徑，並在新檔名加上時戳
        uploaded = FileConfig()
        if not uploaded.change_filename_with_timestamp(file.filename, "to_organized"):
            msg = "整理圖檔--命名不成功"
            res = {
                "status": "error",
                "data": "",
                "msg": msg
            }
            return res
        
        #--儲存整理圖檔到檔案夾
        if uploaded.filepath_organized:
            with open(uploaded.filepath_organized, "wb") as f:
                f.write(content)
            #--釋放上載檔案資源
            await file.close()
        
        #--檢查是否成功儲存
        if uploaded.filepath_organized.exists():
            msg = "整理圖檔--上傳成功"
            res = {
                "status": "success",
                "data": uploaded,       #--回傳FileConfig實例，self的所有屬性(變數)皆可調用
                "msg": msg
            }
            return res
        else:
            msg = "整理圖檔--上傳不成功"
            res = {
                "status": "error",
                "data": "",
                "msg": msg
            }
            return res    
