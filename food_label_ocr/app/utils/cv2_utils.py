"""
OpenCV工具模組
負責影像前處理、輪廓偵測與裁切等 OpenCV 相關功能
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

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
        res = {
            "yx": [[y_start, x_start], [y_end, x_end]],
            "image": image[y_start:y_end, x_start:x_end]
        }
        
        return res
    
    @staticmethod
    def correct_skewed_contour(image: np.ndarray, contour: np.ndarray, padding: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        校正歪斜輪廓，使用透視變換將其平鋪成矩形
        Args:
            image: 原始影像
            contour: 輪廓 (可能歪斜)
            padding: 邊界padding
        Returns:
            (corrected_image, corrected_contour, info_dict): 
            - corrected_image: 校正後的影像
            - corrected_contour: 校正後的輪廓 (標準矩形)
            - info_dict: 包含變換資訊和座標資訊
        """
        # 計算輪廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 使用輪廓逼近取得近似的四邊形頂點
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果逼近結果不是四邊形，使用外接矩形的四個角
        if len(approx) != 4:
            # 取得外接矩形的四個頂點
            src_points = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)
        else:
            # 使用逼近的四邊形頂點，並進行排序
            src_points = approx.reshape(4, 2).astype(np.float32)
            # 對點進行排序：左上、右上、右下、左上
            src_points = OpenCVManager._order_points(src_points)
        
        # 計算目標矩形的寬高
        dst_width = int(w)
        dst_height = int(h)
        
        # 定義目標矩形的四個頂點 (標準位置：左上、右上、右下、左下)
        dst_points = np.array([
            [0, 0],
            [dst_width, 0],
            [dst_width, dst_height],
            [0, dst_height]
        ], dtype=np.float32)
        
        # 計算透視變換矩陣
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 執行透視變換
        corrected_image = cv2.warpPerspective(image, matrix, (dst_width, dst_height))
        
        # 校正後的輪廓是標準矩形
        corrected_contour = np.array([
            [[0, 0]],
            [[dst_width, 0]],
            [[dst_width, dst_height]],
            [[0, dst_height]]
        ], dtype=np.int32)
        
        # 處理邊界和padding
        h_img, w_img = image.shape[:2]
        y_start = max(0, y - padding)
        y_end = min(h_img, y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(w_img, x + w + padding)
        
        info_dict = {
            "original_bbox": [x, y, w, h],
            "bbox_with_padding": [[y_start, x_start], [y_end, x_end]],
            "src_points": src_points.tolist(),
            "dst_points": dst_points.tolist(),
            "perspective_matrix": matrix.tolist(),
            "corrected_size": (dst_width, dst_height)
        }
        
        return corrected_image, corrected_contour, info_dict
    
    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        """
        對四邊形頂點進行排序，確保順序為：左上、右上、右下、左下
        Args:
            points: 四個頂點座標 (4, 2)
        Returns:
            排序後的頂點座標
        """
        # 計算四邊形的中心
        center = points.mean(axis=0)
        
        # 計算每個點相對於中心的角度
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        
        # 按角度排序
        sorted_indices = np.argsort(angles)
        ordered_points = points[sorted_indices]
        
        # 確保順序為左上、右上、右下、左下
        # 計算每個點到左上角(0,0)的距離
        distances_to_tl = np.sqrt(ordered_points[:, 0]**2 + ordered_points[:, 1]**2)
        
        # 找到最接近左上角的點
        tl_idx = np.argmin(distances_to_tl)
        
        # 旋轉數組，使左上角成為第一個
        ordered_points = np.roll(ordered_points, -tl_idx, axis=0)
        
        return ordered_points
