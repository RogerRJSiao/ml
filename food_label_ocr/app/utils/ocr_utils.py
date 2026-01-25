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
    def detect_and_crop_area(image_path: str, saving_dir: str, interest_texts: str) -> bool:
        """
        辨別邊框輪廓後，判斷內部有無指定文字，裁切ROI區域並儲存新圖(OpenCV的AOI)
        Args:
            image_path (str): 原始圖片路徑
            saving_dir (str): 裁切後圖片儲存路徑
        Returns:
            bool: ROI截圖是否儲存成功

        ##--透過AOI取得矩形方法
        方法一. 
        方法二. 轉灰階並模糊化，使用邊緣檢測(Canny)，膨脹邊緣連接斷裂部分，尋找輪廓並選取外框
        方法三. 使用霍夫變換(Hough Transform)偵測直線，計算交點形成矩形，選取最大矩形區域
        """
        #--取得原圖路徑
        if not Path(image_path).exists():
            print(f"圖片路徑不存在: {image_path}")
            return False
        ori_image_path = Path(image_path)

        #--設定輸出目錄、檔案名稱
        img_name = ""
        img_dict = {"desc": "", "path": "", "data": None}
        img_group = {}

        #--讀取原圖資料
        ori_img = cv2.imread(ori_image_path)
        img = ori_img.copy()
        cols, rows = img.shape[1], img.shape[0]

        #--儲存原圖
        img_name, img_dict["desc"] = "img_00", "原始圖檔"
        img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        img_group[img_name] = img_dict.copy()
        color = ori_img.copy()
        cv2.imwrite(img_group[img_name]["path"], color)

        #--將彩圖轉成灰階，提高線條偵測率
        img_name, img_dict["desc"] = "img_01", "灰階圖檔"
        img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        img_group[img_name] = img_dict.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(img_group[img_name]["path"], gray)
        
        #--二值化圖檔(自適應閾值)
        img_name, img_dict["desc"] = "img_02_a", "二值化圖檔-自適應閾值"
        img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        img_group[img_name] = img_dict.copy()
        binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        cv2.imwrite(img_group[img_name]["path"], binary_adaptive)
        #--二值化圖檔(固定閾值)
        img_name, img_dict["desc"] = "img_02_b", "二值化圖檔-固定閾值"
        img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        img_group[img_name] = img_dict.copy()
        binary_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cv2.imwrite(img_group[img_name]["path"], binary_thresh)

        #--標註輪廓(取自自適應二值化，有過濾太小雜訊及原圖邊框)
        img_name, img_dict["desc"] = "img_03_a", "原圖加上輪廓-自適應二值化"
        img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        img_group[img_name] = img_dict.copy()
        contours, _ = cv2.findContours(binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = color.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            #--過濾太小的雜訊方框，以及原圖的邊框
            if (w > 100 and h > 100) and (w < ori_img.shape[1] * 0.9 and h < ori_img.shape[0] * 0.9): 
                cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(img_group[img_name]["path"], contour_img)
        #--標註輪廓(取自固定閾值二值化，有過濾太小雜訊及原圖邊框)
        img_name, img_dict["desc"] = "img_03_b", "原圖加上輪廓-固定閾值二值化"
        img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        img_group[img_name] = img_dict.copy()
        contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img2 = color.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (w > 100 and h > 100) and (w < ori_img.shape[1] * 0.9 and h < ori_img.shape[0] * 0.9): 
                cv2.rectangle(contour_img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(img_group[img_name]["path"], contour_img2)
        
        """
        #--從二值化抽出水平與垂直線條
        img_name, img_dict["desc"] = "img_03_a1", "水平線條圖檔"
        img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        img_group[img_name] = img_dict.copy()
        horizontal = np.copy(binary)
        cols = horizontal.shape[1]
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // 25, 1)) #--增大水平結構元素(3)
        horizontal = cv2.erode(horizontal, h_kernel)
        horizontal = cv2.dilate(horizontal, h_kernel)
        cv2.imwrite(img_group[img_name]["path"], horizontal)

        img_name, img_dict["desc"] = "img_03_a2", "垂直線條圖檔"
        img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        img_group[img_name] = img_dict.copy()
        vertical = np.copy(binary)
        rows = vertical.shape[0]
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // 25))
        vertical = cv2.erode(vertical, v_kernel)
        vertical = cv2.dilate(vertical, v_kernel)
        cv2.imwrite(img_group[img_name]["path"], vertical)

        img_name, img_dict["desc"] = "img_03_a3", "水平垂直線條圖檔"
        img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        img_group[img_name] = img_dict.copy()
        table_mask = cv2.add(horizontal, vertical)
        cv2.imwrite(img_group[img_name]["path"], table_mask)
        """
        # #--從二值化以霍夫變換偵測線段
        # img_name, img_dict["desc"] = "img_03_b1", "霍夫線段檢測圖檔"
        # img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        # img_group[img_name] = img_dict.copy()
        # # threshold: 門檻值, minLineLength: 線段最小長度, maxLineGap: 容許斷裂間距
        # rows, cols = binary.shape[:2]
        # lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=20, 
        #                         minLineLength=20, maxLineGap=20)
        # line_mask = np.zeros_like(binary)   #--建立空白圖
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         # 計算線段角度，過濾掉太短或雜亂的線，只留偏水平與偏垂直的
        #         angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        #         if angle < 20 or angle > 70: # 容許正負 20 度的歪斜
        #             cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
        # # 現在的 line_mask 就是包含斜線的表格骨架，再做後續的矩形提取
        # cv2.imwrite(img_group[img_name]["path"], line_mask)

        # 現在 clean_lines 上的線段更扎實了，再做一次輪廓搜尋
        # contours, _ = cv2.findContours(clean_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     # 過濾掉太小的部分，剩下的就是你要的 ROI
        #     if w > 100 and h > 100: 
        #         print(f"找到 ROI: x={x}, y={y}, w={w}, h={h}")
        # # 畫出矩形區域
        # cv2.rectangle(color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imwrite(img_group[img_name]["path"], color)

        # #--尋找矩形輪廓
        # img_name, img_dict["desc"] = "img_04", "檢測矩形輪廓圖檔"
        # img_dict["path"] = str(Path(saving_dir) / f"{img_name}.jpg")
        # img_group[img_name] = img_dict.copy()
        # contour_line = color.copy()

        # #--顯示原圖大小
        # print(f"原圖大小: {img.shape[1]}x{img.shape[0]}")
        # # 1. 稍微膨脹讓斷裂的線接起來
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        # dilated = cv2.dilate(table_mask, kernel, iterations=2)
        # # 使用較大的 Kernel，例如 7x7 或 9x9
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        # closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite(img_group[img_name]["path"], closed)
        
        # # 2. 尋找輪廓
        # # RETR_EXTERNAL 只找最外框，RETR_TREE 會找所有內層格子
        # contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # rects = []
        # for cnt in contours:
        #     # 計算周長
        #     peri = cv2.arcLength(cnt, True)
        #     # 多邊形擬合 (epsilone 設為周長的 2%-5%)
        #     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
        #     # 如果擬合結果有 4 個頂點，則判定為矩形
        #     if len(approx) == 4:
        #         x, y, w, h = cv2.boundingRect(approx)
        #         print(f"Detected rectangle at x:{x}, y:{y}, w:{w}, h:{h}")
                
        #         #--過濾太小的雜訊方框，以及原圖的邊框
        #         if (w > 40 and h > 20) and (w < img.shape[1]*0.9 and h < img.shape[0]*0.9):
        #             rects.append((x, y, w, h))
        #             #--調用原圖畫上矩形框(綠色)
        #             # cv2.drawContours(contour_line, [approx], -1, (0, 255, 0), 2)
        #             cv2.rectangle(contour_line, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # # rects 現在包含了所有檢測到的矩形坐標 (x, y, w, h)
        # cv2.imwrite(img_group[img_name]["path"], contour_line)
        
        # #--計算線條密度(解決格線不對齊問題)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        # density = cv2.dilate(table_mask, kernel)
        # intersections = cv2.bitwise_and(horizontal, vertical)

        # contours, _ = cv2.findContours(
        #     density,
        #     cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_SIMPLE
        # )
        # adjust_dir_path = Path(adjust_dir) / "image_05.jpg"
        # cv2.imwrite(adjust_dir_path, table_mask)

        # #--找面積最大的矩形輪廓
        # table_cnt = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(table_cnt)
        # table_roi = img[y:y+h, x:x+w]
        # adjust_dir_path = Path(adjust_dir) / "image_06.jpg"
        # cv2.imwrite(adjust_dir_path, table_roi)

        return True