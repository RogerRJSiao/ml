"""
OCR 處理服務
負責模擬 OCR 分析圖片的業務邏輯（實際上複製檔案）
"""
import shutil
from pathlib import Path
from fastapi import HTTPException
from app.utils.file_utils import FileConfig


class OCRProcessService:
    """OCR 處理服務類別"""
    
    @staticmethod
    async def process_ocr(filepath: str) -> dict:
        """
        模擬 OCR 處理流程
        實際操作：複製原始圖片到 annotated 目錄
        
        Args:
            file_config (FileConfig): 包含檔案配置的實例
            
        Returns:
            dict: 包含處理狀態、檔案路徑、OCR 結果的字典
        """

        #--取出待處理的路徑、檔名
        filepath = Path(filepath)
        filename = filepath.name
        #--初始化變數
        res = {}

        #--檢查原圖是否已存在
        if not filepath or not filepath.exists():
            msg = "原圖不存在，無法進行OCR讀圖處理"
            res = {
                "status": "error",
                "data": FileConfig().to_dict(),
                "msg": msg
            }
            return res

        #--確保整理(上載)、標註(下載)目錄存在
        FileConfig.get_organized_dir()
        FileConfig.get_annotated_dir()

        #--準備標註檔案路徑
        annotated = FileConfig()
        if not annotated.change_filename_with_timestamp(filename, "to_organized"):
            msg = "讀取圖檔--命名不成功"
            res = {
                "status": "error",
                "data": annotated.to_dict(),
                "msg": msg
            }
            return res
        if not annotated.change_filename_with_timestamp(filename, "to_annotated"):
            msg = "標註圖檔--命名不成功"
            res = {
                "status": "error",
                "data": annotated.to_dict(),
                "msg": msg
            }
            return res

        #--複製檔案(模擬OCR處理完成)
        shutil.copy2(annotated.filepath_organized, annotated.filepath_annotated)
        
        #--模擬OCR結果
        ocr_result = {
            "status": "success",
            "data": {
                "organized_path": str(annotated.filepath_organized),
                "annotated_path": str(annotated.filepath_annotated),
                "ocr_content": {
                    "text": "食品標示文字內容（模擬）",
                    "confidence": 0.95,
                    "detected_ingredients": ["成分1", "成分2"],
                    "allergens": ["過敏原1"],
                    "nutrition": {
                        "calories": 100,
                        "protein": 5,
                        "fat": 3,
                        "carbs": 15
                    }
                }
            },
            "msg": "OCR 處理完成"
        }
        
        return ocr_result