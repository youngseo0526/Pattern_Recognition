import cv2  # 영상 읽어서 창에 띄우기
import matplotlib.pyplot as plt  # 그래프 그리기

# 이미지 import
img = cv2.imread('lenna.jpeg')

# grayscale 변환
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# histogram 그리기
# cv2.calHist(images(배열), channels(1채널이면 [0], 3채널이면 [0],[1],[0,2]...), mask, histSize(x축 간격), ranges)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])


plt.plot(hist)
plt.show()