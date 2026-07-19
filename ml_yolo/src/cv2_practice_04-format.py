import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

img = cv2.imread("../input/map.jpg")
height, width = img.shape[:2]

#--影像幾何變形/校正--#

#--影像平移
"""
#--平移矩陣 M = [[1,0,tx],[0,1,ty]]，tx/ty分別代表x、y方向位移量(像素)
"""
tx, ty = 100, 50
M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
img_translated = cv2.warpAffine(img, M_translate, (width, height))

cv2.imshow("translate", img_translated)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--影像旋轉
"""
#--cv2.rotate()只能做90度倍數旋轉，任意角度旋轉需用getRotationMatrix2D+warpAffine；
#--getRotationMatrix2D(center, angle, scale)：以center為軸心逆時針旋轉angle度，scale可同時縮放。
"""
center = (width // 2, height // 2)
angle = 30
M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
img_rotated = cv2.warpAffine(img, M_rotate, (width, height))

cv2.imshow("rotate_arbitrary_angle", img_rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--圖像仿射變化、影像透視
"""
#--cv2.getAffineTransform(pts1, pts2)：用3組對應點算出仿射矩陣；
#--cv2.getPerspectiveTransform(pts1, pts2)：用4組對應點算出透視矩陣；
#--仿射變換：3點決定，平行線不變，適合旋轉/平移/縮放/輕微傾斜
#--透視變換：4點決定，平行線可以變不平行，適合校正「拍攝角度造成的變形」(把梯形拉回長方形)
#--
#--變換類型  矩陣大小  可自由設定的參數        需要對應點數
#--仿射      2×3      6個(全部都自由)         3組(6條方程式)
#--透視      3×3      8個(固定1個基準值)      4組(8條方程式)
#--
#--自由度數量本質上就是「矩陣裡有幾個獨立未知數」，
#--而需要的對應點數 = 自由度數 ÷ 2 (因為每個點提供x、y兩條方程式)。
"""
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, M_affine, (width, height))

cv2.imshow("affine", img_affine)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--影像透視
pts1_persp = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
pts2_persp = np.float32([[50, 30], [width - 30, 0], [0, height], [width, height - 40]])
M_persp = cv2.getPerspectiveTransform(pts1_persp, pts2_persp)
img_persp = cv2.warpPerspective(img, M_persp, (width, height))

cv2.imshow("perspective", img_persp)
cv2.waitKey(0)
cv2.destroyAllWindows()


#--圖像標註--#

#--直線、箭頭
img_annotated = img.copy()
cv2.line(img_annotated, (50, 50), (300, 50), (0, 255, 0), 3)          #--綠色直線
cv2.arrowedLine(img_annotated, (50, 100), (300, 100), (0, 0, 255), 3)  #--紅色箭頭

#--四邊形、圓形
cv2.rectangle(img_annotated, (50, 150), (300, 300), (255, 0, 0), 2)   #--藍色矩形框(厚度2)
cv2.circle(img_annotated, (400, 225), 60, (0, 255, 255), -1)          #--黃色實心圓(厚度-1代表填滿)

#--文字(英文可直接用cv2.putText)
cv2.putText(img_annotated, "Hello OpenCV", (50, 350),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#--文字(中文)
"""
#--cv2.putText()不支援中文字型(會顯示問號或亂碼)，需改用PIL繪製後再轉回OpenCV格式：
#--流程：BGR(numpy) -> RGB(PIL Image) -> 用PIL ImageDraw寫中文字 -> 轉回BGR(numpy)
"""
img_pil = Image.fromarray(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)
font = ImageFont.truetype("../resources/font/msjh.ttc", 40)
draw.text((50, 400), "中文標註測試", font=font, fill=(0, 0, 0))
img_annotated = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

cv2.imshow("annotated", img_annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
