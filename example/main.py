import cv2

img = cv2.imread('Lenna.png')

cv2.imshow('image', img)
cv2.waitKey()  # 특정 키를 누르기 전까지 이미지로드를 종료시키지 않음

