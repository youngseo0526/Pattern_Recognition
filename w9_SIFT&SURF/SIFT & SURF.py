import cv2
import time


filepath = 'butterfly.png'
img = cv2.imread(filepath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()  # SIFT 검출기 생성
surf = cv2.xfeatures2d.SURF_create()  # SURF 검출기 생성

for i in [0.5, 1.0, 1.5, 2.0 ]:
    resize = cv2.resize(gray, None, fx=i, fy=i)
    print(">>", resize.shape)

    sift_time = time.time()  # SIFT 시작 시간 저장
    kpts = sift.detect(image=gray, mask=None)  # SIFT keypoints 검출
    print("SIFT :", time.time() - sift_time)  # 실행 시간

    surf_time = time.time()  # SURF 시작 시간 저장
    kpts = surf.detect(image=gray, mask=None)  # SURF keypoints 검출
    print("SURF :", time.time() - surf_time)  # 실행 시간
    print("\n")