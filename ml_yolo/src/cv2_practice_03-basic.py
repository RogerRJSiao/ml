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

#--影像尺寸變更：cv2.resize(img, dsize, fx, fy, interpolation)
"""
#--指定明確寬高(dsize)，或用fx/fy按比例縮放(此時dsize設為None)
#--放大用INTER_CUBIC/INTER_LINEAR，縮小用INTER_AREA，效果較好
"""
img_resize_fixed = cv2.resize(img, (300, 200))  #--直接指定(寬,高)
img_resize_half = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)   #--縮小為50%
img_resize_double = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC) #--放大為200%

cv2.imshow("resize_fixed", img_resize_fixed)
cv2.imshow("resize_half", img_resize_half)
cv2.imshow("resize_double", img_resize_double)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--影像翻轉：cv2.flip(img, flipCode)
"""
#--flipCode  0：上下翻轉(沿x軸)
#--flipCode  1：左右翻轉(沿y軸)
#--flipCode -1：上下+左右都翻轉(180度)
"""
img_flip_v = cv2.flip(img, 0)   #--上下翻轉
img_flip_h = cv2.flip(img, 1)   #--左右翻轉
img_flip_vh = cv2.flip(img, -1) #--上下+左右翻轉

cv2.imshow("flip_vertical", img_flip_v)
cv2.imshow("flip_horizontal", img_flip_h)
cv2.imshow("flip_both", img_flip_vh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--影像旋轉：cv2.rotate(img, rotateCode)
"""
#--只能做90度的倍數旋轉(90/180/270)；比transpose()只能逆時針轉90度，有更多選擇；
#--若需要任意角度旋轉，才需改用cv2.getRotationMatrix2D + cv2.warpAffine。
"""
img_rotate_90cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)          #--順時針轉90度
img_rotate_180 = cv2.rotate(img, cv2.ROTATE_180)                    #--轉180度
img_rotate_90ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  #--逆時針轉90度

cv2.imshow("rotate_90_clockwise", img_rotate_90cw)
cv2.imshow("rotate_180", img_rotate_180)
cv2.imshow("rotate_90_counterclockwise", img_rotate_90ccw)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--影像裁剪(crop)：用numpy切片 img[y1:y2, x1:x2]
"""
#--切片順序是先「高(row/y)」再「寬(col/x)」，跟img.shape的(高,寬,通道數)順序一致；
#--切出來的是原陣列的view(共用記憶體)，若要修改切片內容而不影響原圖，需搭配.copy()。
"""
#--影像裁剪
height, width = img.shape[:2]
x_start, x_end = int(width * 0.25), int(width * 0.75)
y_start, y_end = int(height * 0.25), int(height * 0.75)
img_crop = img[y_start:y_end, x_start:x_end]  #--裁切中央50%區域

#--把裁出來的截圖，貼到與原圖同尺寸的全黑畫布中間
canvas = np.zeros((height, width, 3), dtype=np.uint8)
crop_h, crop_w = img_crop.shape[:2]
paste_x = (width - crop_w) // 2
paste_y = (height - crop_h) // 2
canvas[paste_y:paste_y + crop_h, paste_x:paste_x + crop_w] = img_crop

cv2.imshow("original", img)
cv2.imshow("crop_center", img_crop)
cv2.imshow("crop_on_black_canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


