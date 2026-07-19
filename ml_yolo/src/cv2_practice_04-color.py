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
    
    # cv2.imshow("original", img)
    # cv2.imshow("mask_green", mask_green)
    # cv2.imshow("green_only", img_green_only)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #--影像疊加、影像相減
    """
    #--疊加 → 想要「合成」兩張影像的內容（加上Logo、多重曝光效果、資料擴增、檢查遮罩效果）
    #--相減 → 想要「找出差異」（動態偵測、工業產品瑕疵檢測、去除背景/雜訊）
    """
    #--影像疊加：cv2.addWeighted(img1, alpha, img2, beta, gamma)
    #--          結果 = img1*alpha + img2*beta + gamma，需先將兩張圖resize成相同尺寸
    img2_src = cv2.imread("../input/inv.jpg")
    img2_resized = cv2.resize(img2_src, (img.shape[1], img.shape[0]))
    img_blend = cv2.addWeighted(img, 0.7, img2_resized, 0.3, 0)  #--70% img + 30% img2

    # cv2.imshow("img1", img)
    # cv2.imshow("img2", img2_resized)
    # cv2.imshow("blend", img_blend)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #--影像相減
    """
    #--cv2.subtract(img1, img2)：dst = saturate(img1-img2)，負值會被截斷為0，
    #--                          有方向性(img1-img2 != img2-img1)，
    #--                          適合已知哪張圖較亮/當基準的情境(如已知背景比前景亮)。
    #--cv2.absdiff(img1, img2)：dst = |img1-img2|，取絕對差，恆為正值，
    #--                          absdiff(A,B) == absdiff(B,A)，
    #--                          不需知道變化方向，是動態偵測/差異比對最常用的方式。
    """
    img_sub = cv2.subtract(img, img2_resized)
    img_absdiff = cv2.absdiff(img, img2_resized)

    # cv2.imshow("subtract", img_sub)
    # cv2.imshow("absdiff", img_absdiff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #--圖檔去背(matting)：灰階+背景，存成透明PNG
    """
    #--1. 硬遮罩
    #-- 轉灰階，用門檻判斷「接近白色」的像素(亮度夠高即視為背景)
    #-- -> 轉BGRA多加alpha通道，把背景區域的alpha設為0(透明)，其餘設為255(不透明)
    #-- -> 存檔副檔名務必為.png，jpg不支援透明度
    #--
    #--2. 漸層遮罩
    #-- 只對gray>200的像素做漸層透明(alpha=255-gray)，而非0/255二值遮罩
    #-- 灰階愈接近255(愈白)，alpha愈接近0(愈透明)；灰階愈低於255，愈不透明，
    #-- 邊緣過渡處會有漸層，比硬遮罩更平滑、較不會有鋸齒邊。
    """
    #--1.去背(硬遮罩)
    img_gray4 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask_bg = cv2.threshold(img_gray4, 200, 255, cv2.THRESH_BINARY)  #--亮度>200視為白色背景

    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img_bgra[:, :, 3] = 255 - mask_bg  #--背景(mask_bg=255)處alpha設為0，其餘設為255

    cv2.imwrite("../output/transparent.png", img_bgra)
    print("saved transparent png to: ../output/transparent.png")

    #--2.去背(漸層遮罩)
    img_bgra_vec = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)    #--複製前3通道，第4通道255(不透明)
    #img_gray4 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      #--Gray = 0.114×B + 0.587×G + 0.299×R
    mask = img_gray4 > 200
    img_bgra_vec[mask, 3] = 255 - img_gray4[mask]           #--只有灰階值判為TRUE的才會調整第4通道，其餘保留

    cv2.imwrite("../output/transparent_gradient.png", img_bgra_vec)
    print("saved gradient transparent png to: ../output/transparent_gradient.png")

    cv2.imshow("mask_bg", img_bgra)
    cv2.imshow("mask_gradient_bg", img_bgra_vec)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #--去背之後，套用新背景色(資料擴增)
    """
    #--資料擴增角度：同一個前景物件，套用多種不同背景色，讓模型訓練時
    #--學習「辨識物件本身」而非「記住特定背景」，藉此提升泛化能力。
    #--公式：輸出 = 前景×alpha + 新背景×(1-alpha)，alpha需正規化到0~1才能當混合權重。
    """
    def composite_on_bg(img, alpha_channel, bg_color):
        bg = np.full(img.shape, bg_color, dtype=np.uint8)
        alpha = alpha_channel / 255.0
        alpha_3ch = cv2.merge([alpha, alpha, alpha])
        result = img[:, :, :3] * alpha_3ch + bg * (1 - alpha_3ch)
        return result.astype(np.uint8)

    #--隨機產生N組背景色，模擬資料擴增流程
    np.random.seed(42)  #--固定種子，方便重現同一批擴增結果
    num_augmentations = 4
    bg_colors = np.random.randint(0, 256, size=(num_augmentations, 3)).tolist()  #--BGR隨機色

    for i, bg_color in enumerate(bg_colors):
        img_aug = composite_on_bg(img, img_bgra[:, :, 3], tuple(bg_color))
        cv2.imwrite(f"../output/augmented_bg_{i}.png", img_aug)
        cv2.imshow(f"augmented_bg_{i} (BGR={bg_color})", img_aug)

    print(f"已產生 {num_augmentations} 張不同背景色的擴增圖片")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


