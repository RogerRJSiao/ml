import cv2
import numpy as np

img_src_path = "../input/map.jpg"
img_des_path = "../output/ttst.png"

img = cv2.imread(img_src_path)  #--以原圖開啟

if img is None:
    print("imread failed: check the file path")
else:
    """
    #--調整色彩空間
        #--存檔、顯示、幾何運算 → 留在 BGR；
        #--只看亮度/邊緣/門檻 → 轉灰階；
        #--要「找特定顏色」且不想被光線影響 → 轉 HSV 或 HLS；
        #--需要透明度 → 轉 BGRA。
    """    
    #--原圖轉換為灰階
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    cv2.imshow("gray", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite(img_des_path, img_gray)
    # print("saved grayscale image to:", img_des_path)

    """
    #--產生負片效果：一種正規化的前處理
    #--當偵測演算法預期「亮物件在暗背景」，但實際影像相反(暗物件在亮背景)時，
    #--先反轉顏色即可沿用同一套門檻化/輪廓偵測邏輯，例如：
    #--(1) 文件/發票OCR：亮背景配深色文字，反轉後可用同套亮物件偵測邏輯
    #--(2) X光/醫療影像：反轉讓暗部異常(裂縫、腫瘤)變亮，較容易被看出或偵測到
    #--(3) 過曝/低對比影像：反轉後把細節移到較敏感的亮度區間，利於邊緣偵測
    """
    #--原圖負片
    #--1.用np陣列廣播
    img_negative = 255 - img  #--每個像素反轉 (等同 cv2.bitwise_not(img))
    cv2.imshow("negative", img_negative)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #--2.用for迴圈逐像素、逐通道反轉 (效率遠低於np廣播，僅供學習理解)
    img_negative_loop = img.copy()
    height, width, channels = img.shape
    for y in range(height): #--局部調整，寫range(int(height/2))
        for x in range(width):
            for c in range(channels):  #--分別處理 B、G、R 三個通道
                img_negative_loop[y, x, c] = 255 - img_negative_loop[y, x, c]
    # cv2.imshow("negative_loop", img_negative_loop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #--調整影像亮度與對比度
    """
    #--1.convertScaleAbs：直接指定α、β，以0為軸心縮放，效能好、寫法簡潔；
    #--2.手刻公式：用-127~127的contrast滑桿換算α、β，並以中間灰階為軸心縮放，
    #--           亮度與對比度效果分得更開，較符合繪圖軟體滑桿的直覺操作。(注意溢位問題)
    #--兩者皆為 g = α·f + β 的線性調整，數值對應得當時結果應一致。
    """
    #--1.cv2.convertScaleAbs
    img_bc_cv2 = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
    #--2.手刻對比度/亮度調整公式(換算成與 alpha=1.5, beta=30 等效的數值)
    #--contrast = 127*(alpha-1) = 127*0.5 = 63.5
    #--brightness = beta + contrast = 30 + 63.5 = 93.5
    brightness = 93.5
    contrast = 63.5
    img_bc_formula = np.int16(img)
    img_bc_formula = img_bc_formula * (contrast/127+1) - contrast + brightness
    img_bc_formula = np.clip(img_bc_formula, 0, 255)
    img_bc_formula = np.uint8(img_bc_formula)

    # cv2.imshow("original", img)
    # cv2.imshow("brightness_contrast_cv2", img_bc_cv2)
    # cv2.imshow("brightness_contrast_formula", img_bc_formula)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #--二值化-範例1：灰階+亮暗校正+二值化
    """
    #--二值化前處理整體流程建議：
    #-- 灰階化 → 縮放到適合解析度(文字高度至少 20~30 px) → 傾斜校正(deskew) 
    #-- → 光照/陰影校正(分區塊比較平均亮度、看標準差，高斯模糊取出光照背景，再用adaptive或除法正規化) → 二值化 
    #-- → 形態學修補
    """
    #--灰階化
    img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #--光照/陰影校正：用高斯模糊估計光照背景，再用除法正規化去除陰影
    #--   原理：原圖 = 反射率 × 光照，光照是乘法性、低頻的亮暗變化，
    #--        高斯模糊(大核)可濾除文字等高頻細節，只留下光照背景估計值；
    #--        原圖/光照 = 反射率，無論在亮處或暗處，同一材質的反射率應為一致，
    illumination = cv2.GaussianBlur(img_gray2, (51, 51), 0)
    img_illum_corrected = cv2.divide(img_gray2, illumination, scale=255)
    #--二值化：對已校正光照的影像用Otsu自動找門檻
    _, img_binary = cv2.threshold(img_illum_corrected, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("gray", img_gray2)
    cv2.imshow("illumination", illumination)
    cv2.imshow("illumination_corrected", img_illum_corrected)
    cv2.imshow("binary", img_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #--二值化-範例2：先用直方圖評估雙峰是否明顯，明顯的話直接灰階+二值化即可
    """
    #--判斷方式：用scipy.signal.find_peaks數峰值數量，
    #--        若剛好2個峰且間距、突出程度足夠，代表雙峰分布明顯，
    #--        此時不需額外做光照/陰影校正，直接Otsu二值化即可有穩定效果。
    """
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    img_gray3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img_gray3], [0], None, [256], [0, 256]).flatten()
    peaks, _ = find_peaks(hist, distance=30, prominence=hist.max() * 0.05)

    #--秀出直方圖，並標示偵測到的峰值位置
    plt.plot(hist)
    plt.plot(peaks, hist[peaks], "x")
    plt.xlabel("灰階值")
    plt.ylabel("像素數量")
    plt.title(f"Histogram (peaks: {len(peaks)})")
    plt.show()

    cv2.imshow("original", img)
    if len(peaks) == 2:
        #--雙峰分布明顯：全域亮暗差異清楚，直接用Otsu自動找全域門檻即可
        print("偵測到雙峰分布，峰值位置:", peaks)
        _, img_binary2 = cv2.threshold(img_gray3, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("gray", img_gray3)
        cv2.imshow("binary_bimodal", img_binary2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        #--非明顯雙峰：通常代表光照不均或局部亮度差異，改用adaptiveThreshold依區域局部平均動態調整門檻
        print("非明顯雙峰分布(峰值數量:", len(peaks), ")，改用adaptiveThreshold")
        img_binary2 = cv2.adaptiveThreshold(img_gray3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, blockSize=11, C=2)
        cv2.imshow("gray", img_gray3)
        cv2.imshow("binary_adaptive", img_binary2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #--二值化-範例3：只取特定顏色(綠色)
    """
    #--用HSV而非灰階做顏色篩選：灰階只有亮度資訊，無法區分「顏色相同但亮度不同」的情況，
    #--HSV的H(色相)通道能代表顏色本身，不受亮度/飽和度影響，適合用cv2.inRange()做顏色遮罩。
    #--綠色在OpenCV的H範圍(0~179)大約落在35~85之間，可依實際影像微調上下界。
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 43, 46])   #--H下界, S下界, V下界
    upper_green = np.array([85, 255, 255]) #--H上界, S上界, V上界
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    #--用遮罩取出原圖中的綠色部分，其餘變黑
    img_green_only = cv2.bitwise_and(img, img, mask=mask_green)
    
    cv2.imshow("original", img)
    cv2.imshow("mask_green", mask_green)
    cv2.imshow("green_only", img_green_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
