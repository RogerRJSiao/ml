"""
OCR 處理服務
負責OCR分析圖片的業務邏輯，整合EasyOCR工具進行文字偵測
"""
import shutil
from pathlib import Path
from fastapi import HTTPException
from app.utils.file_utils import FileConfig
from app.utils.ocr_utils import EasyOCRManager

class OCRProcessService:
    """OCR 處理服務類別"""
    
    # 支援的語言代碼 (直接使用 EasyOCR 格式)
    SUPPORTED_LANGUAGES = ["ch_tra", "en", "ja"]
    
    # 包裝樣式映射：前端變數 → readtext 參數配置
    PACKAGING_CONFIG = {
        "normal": {
            "text_threshold": 0.7,
            "low_text": 0.4,
            "description": "一般標準設定"
        },
        "reflective": {
            "text_threshold": 0.5,
            "low_text": 0.3,
            "description": "降低閾值以處理高反光包裝"
        },
        "white_text": {
            "text_threshold": 0.8,
            "low_text": 0.5,
            "description": "提高閾值以處理高對比白色文字"
        },
        "auto": {
            "text_threshold": 0.65,
            "low_text": 0.35,
            "description": "自動中等設定"
        }
    }
    
    @staticmethod
    async def process_ocr(filepath: str, language: str = "ch_tra", packaging: str = "normal") -> dict:
        """
        執行 OCR 處理流程，整合 EasyOCR 進行文字偵測與信心度判斷
        
        Args:
        - filepath (str): 原始圖檔路徑
        - language (str): 辨識語言代碼 (traditional_chinese/english/japanese)
        - packaging (str): 包裝樣式 (normal/reflective/white_text/auto)
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

        #--驗證輸入參數，若不符則使用預設值
        if language not in OCRProcessService.SUPPORTED_LANGUAGES or language == "ch_tra" :
            langs_list = ["ch_tra", "en"]  #--預設繁體中文+英文     #--預設繁體中文
        else:
            langs_list = [language]
        if packaging not in OCRProcessService.PACKAGING_CONFIG:
            packaging = "normal"    #--預設一般

        #--確保整理(上載)、調整後、標註(下載)目錄存在
        FileConfig.get_organized_dir()
        FileConfig.get_adjusted_dir()
        FileConfig.get_annotated_dir()

        #--準備顯示結果檔案路徑
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
        
        #--建立調整後的檔案路徑(用時戳命名)
        adjust_dirpath = FileConfig.ADJUSTED_DIR / f"adj_{annotated.timestamp_str}"
        annotated.dirpath_adjusted = adjust_dirpath
        if not adjust_dirpath.exists():
            adjust_dirpath.mkdir(parents=True, exist_ok=True)

        #--分割營養標示表區域
        cropped_result = EasyOCRManager.detect_and_crop_area(
            image_path=str(annotated.filepath_organized),
            saving_dir=str(annotated.dirpath_adjusted),
            interest_texts="標示"
        )
         
        #--執行並存取EasyOCR文字偵測結果
        ocr_result = EasyOCRManager.detect_text(
            image_path=str(annotated.filepath_organized),
            languages=langs_list,
            packaging_style=packaging,
            gpu=True  #--預設嘗試使用GPU，如不可用則自動切換成CPU
        )
        
        #--準備OCR回傳內容
        ocr_content = {}
        if ocr_result["status"] == "success":
            processed_results = ocr_result["data"]["processed_results"]
            
            #--所有檢測到的文字
            all_detected_text = EasyOCRManager.get_detected_text(processed_results)
            
            ocr_content = {
                "status": "success",
                "detected_text": all_detected_text,
                "total_detections": len(processed_results),
                "raw_results": processed_results
            }
        else:
            # OCR 偵測失敗，回傳錯誤訊息
            ocr_content = {
                "status": "error",
                "message": ocr_result["message"]
            }

        #--獲取包裝樣式的實際配置
        packaging_config = OCRProcessService.PACKAGING_CONFIG[packaging]
        
        #--組合最終回傳結果
        ocr_result = {
            "status": "success",
            "data": {
                "organized_path": str(annotated.filepath_organized),
                "annotated_path": str(annotated.filepath_annotated),
                "processing_config": {
                    "language": language,
                    "packaging": packaging,
                    "packaging_config": {
                        "text_threshold": packaging_config["text_threshold"],
                        "low_text": packaging_config["low_text"],
                        "description": packaging_config["description"]
                    }
                },
                "ocr_content": ocr_content
            },
            "msg": f"OCR 處理完成 (指定語言: {language}, 指定樣式: {packaging})"
        }
        
        return ocr_result
