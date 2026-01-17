"""
檔案工具模組
負責檔案儲存、檢查、安全命名及路徑管理
| 實例方法 | 第一參數self | 有機會異動「這個物件」資料 |
| 類別方法 | 第一參數cls  | 有機會異動「這個類別」資料 | 取得允許的副檔名列表、驗證檔案副檔名
| 靜態方法 | 無參數       | 純工具，不用類別或實例     | 字串驗證工具
"""
from pathlib import Path
from typing import Set, Tuple
import os
import time

class FileConfig:
    """檔案配置類別 - 管理上傳和標註圖片目錄"""
    #--設定上載、標註的目錄路徑
    UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
    ORGANIZED_DIR = UPLOAD_DIR / "organized"    # 原始圖片
    ANNOTATED_DIR = UPLOAD_DIR / "annotated"    # 標註圖片

    def __init__(self):
        """初始化實例屬性"""
        self.timestamp_str = ""
        self.filename_original = ""
        self.filepath_organized = ""
        self.filepath_annotated = ""
    
    def set_timestamp_str(self) -> str:
        """取得當前時間戳字串，格式: YYYYMMDD_HHMMSS_ms"""
        t = time.time()
        local_time = time.localtime(t)
        time_str = time.strftime("%Y%m%d_%H%M%S", local_time)
        ms = int((t - int(t)) * 10000)
        
        #--組合完整時間戳字串(西元年到毫秒)
        self.timestamp_str = f"{time_str}_{ms}"
        return self.timestamp_str

    def change_filename_with_timestamp(self, filename: str, process: str) -> bool:
        """
        根據當前時間戳修改檔案名稱
        
        Args:
            filename (str): 原始檔案名稱
            process (str): 處理類型 ('to_organized' 或 'to_annotate')
            
        Returns:
            bool: 成功返回 True，失敗返回 False
        """
        #--取得預計要改前的路徑、副檔名
        # filename = os.path.basename(filepath) #--只取檔名+副檔名部分
        stem, ext = os.path.splitext(filename)  #--ext從最後一點分割(包含.)
        #--檢查副檔名是否存在且允許
        if not ext or ext.lower() not in FileValidator.ALLOWED_EXTENSIONS:
            return False

        #--組合新檔名
        if process == "to_organized":
            #--存取原始檔名
            self.filename_original = filename            
            #--儲存原始圖檔用
            if not self.timestamp_str:
                self.set_timestamp_str()
            self.filepath_organized = self.__class__.ORGANIZED_DIR / f"up_{self.timestamp_str}{ext}"
        
        elif process == "to_annotated":
            #--儲存標註圖檔用
            if not self.timestamp_str:
                return False
            self.filepath_annotated = self.__class__.ANNOTATED_DIR / f"tag_{self.timestamp_str}{ext}"
        
        else:
            return False
            
        return True
    
    @classmethod
    def get_organized_dir(cls) -> Path:
        """確認或建立原始圖片目錄"""
        cls.ORGANIZED_DIR.mkdir(parents=True, exist_ok=True)
        return cls.ORGANIZED_DIR
    
    @classmethod
    def get_annotated_dir(cls) -> Path:
        """確認或建立標註圖片目錄"""
        cls.ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        return cls.ANNOTATED_DIR


class FileValidator:
    """檔案驗證類別"""
    #--允許的副檔名集合，檢查上傳檔案是否符合要求
    #--Set是不重複集合，可用於in快速查找，時間複雜度O(1)較list的O(n)快
    #--改用frozenset({})，不可變集合
    ALLOWED_EXTENSIONS: Set[str] = {
        ".jpg", ".jpeg", ".png", ".webp", ".heic"
    }
    
    @classmethod
    def validate_extension(cls, filename: str) -> Tuple[bool, str]:
        """
        驗證圖檔副檔名
        Args:       filename (str): 要驗證的檔案名稱
        Returns:    (Tuple[bool, str]): (是否有效, 錯誤訊息)
        """
        #--取得副檔名並轉小寫
        file_ext = Path(filename).suffix.lower()
        #--副檔名原本不存在
        if not file_ext:
            return False, "檔案必須有副檔名"
        #--副檔名不在允許列表中
        if file_ext not in cls.ALLOWED_EXTENSIONS:
            allowed = cls.get_allowed_extensions()
            return False, f"您上載了不允許的圖檔類型。指定：{allowed}"
        
        return True, ""
    
    @classmethod
    def get_allowed_extensions(cls) -> str:
        """獲取允許的副檔名列表（逗號分隔）"""
        return ", ".join(sorted(cls.ALLOWED_EXTENSIONS))
