import cv2
import numpy as np

# 모델 영상의 2차원 H-S 히스토그램 계산
img_m = cv2.imread('model.jpg')
hsv_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2HSV)
hist_m = cv2.calcHist([hsv_m], [0, 1], None, [180, 256], [0, 180, 0, 256])

# 입력 영상의 2차원 H-S 히스토그램 계산
img_i = cv2.imread('hand.jpg')
hsv_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2HSV)
hist_i = cv2.calcHist([hsv_i], [0, 1], None, [180, 256], [0, 180, 0, 256])

# 히스토그램 정규화
##과제
# height_m, width_m = img_m.shape[0], img_m.shape[1]
# height_i, width_i = img_i.shape[0], img_i.shape[1]
hist_m = hist_m / (img_m.shape[0] * img_m.shape[1])  # (height_m*width_m)=img_m.size
hist_i = hist_i / img_i.size  # (height_i*width_i)=img_i.size
##
print("maximum of hist_m: %f" % hist_m.max())  # 값 범위 체크 :1.0 이하
print("maximum of hist_i: %f" % hist_i.max())  # 값 범위 체크 :1.0 이하

# 비율 히스토그램 계산 (신뢰도 함수)
##과제
hist_r = hist_m / (hist_i + 1e-7)
hist_r = np.minimum(hist_r, 1.0)
##
print("range of hist_r: [%.1f, %.1f]" % (hist_r.min(), hist_r.max()))  # 비율 값 범위 체크

# 히스토그램 역투영 수행
##과제
print(img_i.shape)
height, width = img_i.shape[0], img_i.shape[1]  # 영상의 높이와 너비 정보
result = np.zeros_like(img_i, dtype='float32')  # 입력 영상과 동일한 크기의 0으로 초기화된 배열 생성
h, s, v = cv2.split(hsv_i)  # 채널 분리

for i in range(height):  # 모든 픽셀을 순회하며 처리
    for j in range(width):
        h_value = h[i, j]  # (i,j)번째 픽셀의 hue값
        s_value = s[i, j]  # (i,j)번째 픽셀의 saturation값
        confidence = hist_r[h_value, s_value]  # (i,j)번째 픽셀의 신뢰도 점수
        result[i, j] = confidence  # 신뢰도 점수를 결과 이미지 (i,j)번째 픽셀에 저장장
##

# 이진화 수행 (화소값이 임계값 0.02보다 크면 255, 그렇지 않으면 0)
ret, thresholded = cv2.threshold(result, 0.02, 255, cv2.THRESH_BINARY)
cv2.imwrite('result.jpg', thresholded)

# 모폴로지 연산 적용
##과제
kernel = np.ones((13, 13), np.uint8)
improved = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
##
cv2.imwrite('morphology.jpg', improved)

'''
- 컬러 이미지 채널 순서
    - matplotlib: R, G, B 순
    - opencv : B, G, R 순
- 이미지 배열의 형태: img.shape -> (height, width, n_channel)
- cv.show(), plt.imshow()를 이용한 이미지 시각화
    - opencv
        - float32: 화소값 범위 [0.0, 1.0]
        - uint8: 화소값 범위 [0, 255]
    - matplotlib
        - 최소값과 최대값이 [0, 255]가 되도록 정규화한 결과 출력
- Numpy 배열 연산 특성
    - broad-casting
        np.array([1,2,3,4])*10 -> np.array([10,20,30,40])
    - 요소별 연산
        np.array([10,20,30,40])/np.array([1,2,3,4]) -> np.array([10,10,10,10])
'''