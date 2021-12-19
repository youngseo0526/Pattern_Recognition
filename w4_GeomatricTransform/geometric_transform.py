import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(17, 5))

img = cv2.imread('hand.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height = img.shape[0]
width = img.shape[1]
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.axis('off')
plt.title('original')


# OpenCV를 이용한 변환 행렬 도출
center = (width / 2, height / 2)
print(center)
cv_M = cv2.getRotationMatrix2D(center, 90, 1.0)  # 회전 방향이 반시계방향(CCW; Counter Clock-Wise) # 양수는 반시계 방향, 음수는 시계 방향
cv_result = cv2.warpAffine(img, cv_M, (width, height))
print('>> OpenCV Rotation matrix')
print(cv_M, end='\n\n')

plt.subplot(1, 3, 2)
plt.imshow(cv_result)
plt.axis('off')
plt.title('cv_result')

# 직접 도출한 행렬을 이용한 회전 변환 #
# 기준을 중심점으로 이동 변환 행렬
M1 = np.array([[1, 0, -center[0]],
                [0, 1, -center[1]],
                [0, 0, 1]])

# 반 시계방향 90도 회전 변환 행렬 
# (openCV는 윗원점 기준, 교수님은 아래 원점 기준 -> openCV에서 반시계는 교수님에서 시계 방향 회전)
M2 = np.array([[0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]])  # sin90 = 1, cos90 = 0

# 기준을 원래의 원점(0,0)으로 이동 변환 행렬
M3 = np.array([[1, 0, center[0]],
                [0, 1, center[1]],
                [0, 0, 1]])  # M1의 전치 행렬

# 단일 변환행렬 생성
tmp = np.matmul(M2, M1)
my_M = np.matmul(M3, tmp)
my_M = my_M[:2, ]  # 실제 연산을 수행하는 warpAffine에 동차 행렬이 아닌 2x3 행렬을 넘겨줘야 함


print('>> My matrix')
print(my_M)
my_result = cv2.warpAffine(img, my_M, (width, height))

plt.subplot(1, 3, 3)
plt.imshow(my_result)
plt.axis('off')
plt.title('my_result')

# figure 출력
plt.tight_layout()
plt.show()

'''
- 정방 변환은 앨리어싱 문제 발생하기 때문에 opencv의 warpAffine 함수 내부에서 후방 기하 변환을 적용해줌
- warpAffine 함수는 보간법으로 bilinear 함수 사용(파라미터 바꿔서 다른 보간법 사용 가능)
'''