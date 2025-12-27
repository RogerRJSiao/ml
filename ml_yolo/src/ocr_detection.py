#--在終端機安裝套件（不要寫進程式內
# pip install opencv-python==4.10.0.84
# pip install easyocr
# pip install torch torchvision torchaudio easyocr
# pip install supervision

import os               
import cv2              
import numpy as np      
import easyocr          
import supervision as sv
from PIL import ImageFont, ImageDraw, Image

#--定義並檢查檔案輸入、輸出路徑
INPUT_DIR = "input"                            
# IMAGE_PATH = os.path.join(INPUT_DIR, "inv.jpg")
IMAGE_PATH = os.path.join(INPUT_DIR, "inv2.jpg")
if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(f"找不到影像檔：{IMAGE_PATH}，請確認路徑是否正確。")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "annoted_inv.jpg")
# if os.path.exists(OUTPUT_PATH):
#     raise FileExistsError(f"輸出檔案已存在：{OUTPUT_PATH}，請避免覆寫。")

#--讀取圖檔
image = cv2.imread(IMAGE_PATH)
#--預處理影像(非必要)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.adaptiveThreshold(
#     gray, 255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     11, 2
# )

#--初始化EasyOCR讀取器，同時檢查有無啟用GPU
lang = ["ch_tra","en"]  #--繁體中文+英文(EasyOCR模型不接受中文+日文)
try:
    #--嘗試強制使用GPU
    reader = easyocr.Reader(lang, gpu=True)
    print("EasyOCR使用GPU")
except Exception as e:
    #--任何 GPU 相關錯誤都會被抓到
    print("GPU不可用，改用CPU執行。")
    print("錯誤訊息：", e)
    #--改用CPU
    reader = easyocr.Reader(lang, gpu=False)    

#--執行文字偵測
#--回傳格式 List[ [bbox(4點), text, confidence], ... ]
result_ocr = reader.readtext(
    IMAGE_PATH,
    detail=1,
    paragraph=False,
    text_threshold=0.7,
    low_text=0.4
)

#--將EasyOCR結果轉成supervision的Detections
#--逐筆取出 OCR 偵測結果(圖框、信心分數、類別ID、標籤文字)
xyxy, confidences, class_ids, labels = [], [], [], []
for bbox, text, confidence in result_ocr:
    #--取得辨識結果的圖框座標
    #--bbox 是 4 個點 (x, y)，這裡轉成左上/右下座標    
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x_min, y_min = int(min(xs)), int(min(ys))   #--左上
    x_max, y_max = int(max(xs)), int(max(ys))   #--右下
    xyxy.append([x_min, y_min, x_max, y_max])
    
    #--取得辨識結果的文字(用作標籤)、信心分數、類別ID(暫先定為0)
    confidences.append(float(confidence))
    class_ids.append(0)
    labels.append(text)

print(labels) 

#--複製影像檔用於標註
annotated = image.copy()
#--檢查是否有偵測結果
if len(xyxy) > 0:
    #--建立Detections物件(資料型別np)
    detections = sv.Detections(
        xyxy        = np.array(xyxy, dtype=np.int32),           #--方框座標陣列（整數）
        confidence  = np.array(confidences, dtype=np.float32),  #--信心分數陣列（浮點數）
        class_id    = np.array(class_ids, dtype=np.int32)       #--類別ID陣列（整數）
    )

    #--建立標註器
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    #--用supervision標註方框
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    #--用supervision標註文字(考量中文顯示問題，改用PIL標註)
    # annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    
    #--用PIL標註繁體中文
    img_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("D:\git_proj\ml\ml_yolo\Font\msjhbd.ttc", 24)   #--取自MS字體檔
    for box, label in zip(detections.xyxy, labels):
        x1, y1, x2, y2 = map(int, box[:4])
        draw.text((x1, y1-25), label, font=font, fill=(0, 0, 255))

    #--將PIL格式變更回OpenCV格式，寫檔
    annotated = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PATH, annotated)
    #--檢查輸出大小
    if os.path.exists(OUTPUT_PATH):
        file_size = os.path.getsize(OUTPUT_PATH)
        print(f"已輸出標註影像：{OUTPUT_PATH}")
        print(f"輸出檔案大小：{file_size / 1024:.2f} KB")

    #--顯示結果影像
    cv2.imshow("EasyOCR Result (Press any key to close)", annotated)
else:   
    #--沒有任何偵測結果
    print("未偵測到任何文字。將顯示原始影像。")
    cv2.imshow("EasyOCR Original (Press any key to close)", image)

#--等待或關閉OpenCV視窗
cv2.waitKey(0)
cv2.destroyAllWindows()