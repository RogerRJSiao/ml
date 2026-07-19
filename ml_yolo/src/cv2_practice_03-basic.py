import cv2
import numpy as np

print("cv2 version:", cv2.__version__)

img_src_path = "../input/map.jpg"
img_des_path = "../output/ttst.png"

img = cv2.imread(img_src_path)  #--以原圖開啟
img_gray = cv2.imread(img_src_path, cv2.IMREAD_GRAYSCALE) #--以灰階開啟

if img is None or img_gray is None:
    print("imread failed: check the file path")
else:
    #--顯示在新視窗
    print("imread succeeded, image shape:", img.shape)  #--圖片屬性-(寬,高,通道數)
    print("size:", img.size)  #--圖片屬性-寬*高*通道數
    print("dtype:", img.dtype)  #--圖片屬性-資料型別
    cv2.imshow("name of window", img)
    cv2.waitKey(0)  #--可設毫秒數自動關閉視窗
    cv2.destroyAllWindows()

    #--取得高和寬
    height, width = img.shape[:2]  #--img.shape為(高,寬,通道數)，取前兩個值
    print("height:", height, "width:", width)

    #--寫入新檔案(灰階, PNG 壓縮品質 80%)
    cv2.imwrite(img_des_path, img_gray, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("saved grayscale image to:", img_des_path)

    #--逐一取出單色，透過切片賦值
    img_pink = img.copy()
    img_pink[:,:,1] = 0  #--去除綠色
    img_blue = img_pink.copy()
    img_blue[:,:,2] = 0  #--去除紅色
    cv2.imshow("img_pink", img_pink)
    cv2.imshow("img_blue", img_blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#--建立三維陣列，儲存BGR自訂圖片
img_np = np.zeros((500, 500, 3), dtype=np.uint8)
#--中央 200*200 px填滿黃色 (BGR: B=0, G=255, R=255)
img_np[150:350, 150:350] = (0, 255, 255)
cv2.imshow("numpy image", img_np)
cv2.waitKey(0)
cv2.destroyAllWindows()

