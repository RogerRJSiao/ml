"""
EasyOCR 工具模組
負責 EasyOCR 初始化、文字偵測、結果處理及信心度判斷
"""

import easyocr
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os
from app.utils.file_utils import FileConfig
from app.utils.cv2_utils import OpenCVManager

class EasyOCRManager:
    """EasyOCR 管理器 - 負責 OCR 相關操作"""
    
    # 全域 Reader 實例 (單例模式避免重複初始化)
    _reader_instances = {}
    
    # 信心度閾值配置
    CONFIDENCE_THRESHOLDS = {
        "normal": {
            "text_threshold": 0.7,
            "low_text": 0.4
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
    def get_reader(languages: List[str], gpu: bool = True) -> easyocr.Reader:
        """
        取得或建立EasyOCR Reader實例 (單例)
        Args:
            languages (List[str]): 語言代碼列表 (e.g., ['ch_tra', 'en'])
            gpu (bool): 是否使用 GPU，預設 True
        Returns:
            easyocr.Reader: Reader 實例
        """
        #--建立唯一鑰匙標識
        lang_key = tuple(sorted(languages))
        instance_key = (lang_key, gpu)
        
        #--若尚未建立則建立新實例
        if instance_key not in EasyOCRManager._reader_instances:
            try:
                reader = easyocr.Reader(languages, gpu=gpu)
                EasyOCRManager._reader_instances[instance_key] = reader
                print("EasyOCR使用GPU模式讀取")
                return reader
            except Exception as e:
                print(f"GPU初始化失敗: {e}")
                if gpu:
                    print("EasyOCR改用CPU模式讀取")
                    reader = easyocr.Reader(languages, gpu=False)
                    EasyOCRManager._reader_instances[instance_key] = reader
                    return reader
                else:
                    raise
        
        return EasyOCRManager._reader_instances[instance_key]
    
    @staticmethod
    def detect_text(
        image_path: str,
        languages: List[str] = ["ch_tra", "en"],    #--繁體中文+英文(EasyOCR模型不接受中文+日文)
        packaging_style: str = "normal",
        gpu: bool = True
    ) -> Dict:
        """
        執行文字偵測並回傳結構化結果
        Args:
            image_path (str): 圖片路徑
            languages (List[str]): 語言代碼列表
            packaging_style (str): 包裝樣式 (normal/reflective/white_text/auto)
            gpu (bool): 是否使用 GPU
        Returns:
            (Dict): 包含偵測結果、信心度統計的字典
        """
        #--驗證圖片是否存在
        image_path = Path(image_path)
        if not image_path.exists():
            return {
                "status": "error",
                "message": f"圖片不存在: {image_path}",
                "data": None
            }
        
        try:
            #--讀取圖片
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    "status": "error",
                    "message": "無法讀取圖片文件",
                    "data": None
                }
            
            #--取得Reader實例
            reader = EasyOCRManager.get_reader(languages, gpu)
            
            # 取得包裝樣式對應的參數
            threshold_config = EasyOCRManager.CONFIDENCE_THRESHOLDS.get(
                packaging_style, 
                EasyOCRManager.CONFIDENCE_THRESHOLDS["normal"]
            )
            
            #--執行OCR偵測
            result_ocr = reader.readtext(
                str(image_path),
                detail=1,
                paragraph=False,
                text_threshold=threshold_config["text_threshold"],
                low_text=threshold_config["low_text"]
            )
            
            #--處理偵測結果，並建立標註影像
            ocr_data = EasyOCRManager._process_ocr_results_and_create_annotated_image(languages, image_path, result_ocr)
            
            return {
                "status": "success",
                "message": "文字偵測完成",
                "data": {
                    "raw_results": result_ocr,
                    "processed_results": ocr_data,
                    "image_shape": image.shape,
                    "languages": languages,
                    "packaging_style": packaging_style,
                    "threshold_config": threshold_config
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"OCR 處理失敗: {str(e)}",
                "data": None
            }
    
    @staticmethod
    def _process_ocr_results_and_create_annotated_image(languages, image_path, result_ocr: List) -> List[Dict]:
        """
        處理 EasyOCR 的原始結果，轉為結構化格式並建立標註圖片
        Args:
            languages: 語言列表
            image_path (str): 圖片路徑
            result_ocr (List): EasyOCR原始偵測結果
        Returns:
            List[Dict]: 處理後的結構化結果
        """
        processed_results = []

        #--準備字型用於標註文字
        if languages:
            font = ImageFont.truetype("D:\git_proj\ml\ml_yolo\Font\msjhbd.ttc", 18)   #--取自MS字體檔

        #--整理寫檔路徑
        organized_image_path = Path(image_path)
        organized_image_name = organized_image_path.name
        annotated_image_name = organized_image_name.replace("up_", "tag_")
        annotated_image_path = FileConfig.ANNOTATED_DIR / annotated_image_name
        #--準備新圖
        organized_image = cv2.imread(image_path)
        annotated_image = organized_image.copy()        
        #--建立PIL圖像用於標註
        img_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        for bbox, text, confidence in result_ocr:
            #--計算邊框座標 (轉換為左上、右下格式)
            #--bbox 是一個點陣列 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            try:
                bbox_list = [[float(p[0]), float(p[1])] for p in bbox]
                xs = [p[0] for p in bbox_list]
                ys = [p[1] for p in bbox_list]
            except (TypeError, IndexError):
                #--如果轉換失敗，跳過此項
                continue
            
            x_min, y_min = int(min(xs)), int(min(ys))
            x_max, y_max = int(max(xs)), int(max(ys))
            # print(f"Detected text: {text} at [{x_min}, {y_min}, {x_max}, {y_max}] with confidence {confidence}")
            
            processed_results.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": {
                    "points": bbox_list,
                    "xyxy": [x_min, y_min, x_max, y_max],
                    "width": x_max - x_min,
                    "height": y_max - y_min
                }
            })

            #--用PIL建立標註
            #--畫邊界框 (紅色矩形)
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=2)
            #--畫文字標籤 (藍色字體)
            draw.text((x_min, y_min-25), text, font=font, fill=(0, 0, 255))
            
        #--將PIL格式變更回OpenCV格式，寫入新檔
        annotated_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imwrite(annotated_image_path, annotated_image)

        #--檢查輸出大小
        if Path(annotated_image_path).exists():
            file_size = os.path.getsize(annotated_image_path)
            print(f"已輸出標註影像：{annotated_image_path}")
            print(f"輸出檔案大小：{file_size / 1024:.2f} KB")
        
        return processed_results
    
    @staticmethod
    def get_detected_text(processed_results: List[Dict]) -> str:
        """
        從處理後的結果中提取所有檢測到的文字
        Args:
            processed_results (List[Dict]): 處理後的 OCR 結果
        Returns:
            str: 合併後的文字 (空格分隔)
        """
        return " ".join([r["text"] for r in processed_results])

    @staticmethod
    def detect_and_crop_area(image_path: str, saving_dir: str, interest_texts: str = "") -> bool:
        """
        辨別邊框輪廓後，裁切ROI區域並儲存處理過程的圖像
        Args:
            image_path (str): 原始圖片路徑
            saving_dir (str): 裁切後圖片儲存路徑
            interest_texts (str): 目標文字（用於後續篩選ROI）
        Returns:
            bool: 處理是否成功
        """
        #--驗證輸入參數
        if not Path(image_path).exists():
            print(f"圖片路徑不存在: {image_path}")
            return False
        saving_dir = Path(saving_dir)
        saving_dir.mkdir(parents=True, exist_ok=True)
        
        #--讀取原圖資料、尺寸
        ori_img = OpenCVManager.read_image(str(image_path))
        if ori_img is None:
            print(f"無法讀取圖片: {image_path}")
            return False
        img_shape = ori_img.shape[:2]  # (height, width)
        
        #--保存原始圖像
        OpenCVManager.save_image(ori_img, str(saving_dir / "img_00_original.jpg"))
        
        #--轉換為灰階
        gray = OpenCVManager.convert_to_gray(ori_img)
        OpenCVManager.save_image(gray, str(saving_dir / "img_01_gray.jpg"))
        
        #--應用自適應二值化
        binary_adaptive = OpenCVManager.apply_adaptive_threshold(gray)
        OpenCVManager.save_image(binary_adaptive, str(saving_dir / "img_02_a_binary_adaptive.jpg"))
        
        #--應用Otsu二值化
        binary_otsu = OpenCVManager.apply_otsu_threshold(gray)
        OpenCVManager.save_image(binary_otsu, str(saving_dir / "img_02_b_binary_otsu.jpg"))
        
        #--使用自適應二值化尋找輪廓
        valid_contours_adaptive = OpenCVManager.find_valid_contours(binary_adaptive, img_shape)
        contour_img_adaptive = OpenCVManager.draw_contours(ori_img, valid_contours_adaptive)
        OpenCVManager.save_image(contour_img_adaptive, str(saving_dir / "img_03_a_contours_adaptive.jpg"))
        
        #--使用Otsu二值化尋找輪廓
        valid_contours_otsu = OpenCVManager.find_valid_contours(binary_otsu, img_shape)
        contour_img_otsu = OpenCVManager.draw_contours(ori_img, valid_contours_otsu)
        OpenCVManager.save_image(contour_img_otsu, str(saving_dir / "img_03_b_contours_otsu.jpg"))

        #--彙整兩種方法的輪廓數量
        print(f"自適應二值化找到有效輪廓數量: {len(valid_contours_adaptive)}")
        print(f"Otsu二值化找到有效輪廓數量: {len(valid_contours_otsu)}")
        contours_to_process = valid_contours_adaptive + valid_contours_otsu
        
        #--裁切ROI區域並儲存
        roi_count = 0        
        for idx, cnt in enumerate(contours_to_process):
            roi = OpenCVManager.crop_roi(ori_img, cnt, padding=5)
            roi_path = saving_dir / f"img_roi_{idx:02d}.jpg"
            OpenCVManager.save_image(roi, str(roi_path))
            roi_count += 1
            
            #--可選：檢測ROI內的文字並篩選
            if interest_texts:
                res = EasyOCRManager.detect_text(str(roi_path), languages=["ch_tra", "en"], packaging_style="normal", gpu=False)
                if res["status"] == "success":
                    detected_text = EasyOCRManager.get_detected_text(res["data"]["processed_results"])
                    if interest_texts in detected_text:
                        print(f"找到目標文字 '{interest_texts}'，已儲存 ROI 圖片：{roi_path}")
        
        print(f"處理完成，共裁切 {roi_count} 個 ROI 區域")
        return True