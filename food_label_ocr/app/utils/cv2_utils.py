"""
OpenCV工具模組
負責影像前處理、輪廓偵測與裁切等 OpenCV 相關功能
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

class OpenCVManager:
    """OpenCV管理器"""
    
    @staticmethod
    def read_image(image_path: str) -> Optional[np.ndarray]:
        """讀取影像"""
        if not Path(image_path).exists():
            return None
        return cv2.imread(str(image_path))

    @staticmethod
    def save_image(image: np.ndarray, save_path: str) -> bool:
        """儲存影像"""
        try:
            cv2.imwrite(str(save_path), image)
            return True
        except Exception as e:
            print(f"儲存影像失敗 {save_path}: {e}")
            return False

    @staticmethod
    def convert_to_gray(image: np.ndarray) -> np.ndarray:
        """轉換為灰階"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply_adaptive_threshold(gray_image: np.ndarray) -> np.ndarray:
        """應用自適應二值化"""
        return cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )

    @staticmethod
    def apply_otsu_threshold(gray_image: np.ndarray) -> np.ndarray:
        """應用 Otsu 二值化"""
        return cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

    @staticmethod
    def find_valid_contours(
        binary_image: np.ndarray, 
        img_shape: Tuple[int, int],
        min_size: int = 100,
        max_ratio: float = 0.9
    ) -> List[np.ndarray]:
        """
        尋找並過濾輪廓
        Args:
            binary_image: 二值化影像
            img_shape: 原圖尺寸 (height, width)
            min_size: 最小邊長
            max_ratio: 最大邊長佔原圖比例
        Returns:
            符合條件的輪廓列表
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        h_img, w_img = img_shape
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 過濾條件: 大於 min_size 且 小於原圖 90%
            if (w > min_size and h > min_size) and (w < w_img * max_ratio and h < h_img * max_ratio):
                valid_contours.append(cnt)
                
        return valid_contours

    @staticmethod
    def draw_contours(image: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
        """在影像上繪製輪廓與邊框"""
        result_img = image.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 255), 2)
        return result_img

    @staticmethod
    def crop_roi(image: np.ndarray, contour: np.ndarray, padding: int = 5) -> np.ndarray:
        """裁切輪廓區域 (含 padding)"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # 處理邊界，避免負值索引導致錯誤裁切
        h_img, w_img = image.shape[:2]
        y_start = max(0, y - padding)
        y_end = min(h_img, y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(w_img, x + w + padding)
        
        return image[y_start:y_end, x_start:x_end]
