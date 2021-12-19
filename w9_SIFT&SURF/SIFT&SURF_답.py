import cv2
import time

sift = cv2.xfeatures2d.SIFT_create()  # SIFT 검출기 생성
surf = cv2.xfeatures2d.SURF_create()  # SURF 검출기 생성

for scale_factor in [0.5, 1.0, 2.0, 10]:
    # 이미지 불러와서 리사이징
    img = cv2.imread('butterfly.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
    # 대상이미지, list 형태로 변환하고자하는 너비/높이정보, 비율적으로 할 때는 2번째 파라미터 None, 3,4번째에 비율 넘김
    print(">>", gray.shape)

    # SIFT 특징 검출 속도 계산
    t1 = time.time()
    keypoints = sift.detect(image=gray, mask=None)
    t2 = time.time()
    print("SIFT: %f sec" % (t2 - t1))

    # SURF 특징 검출 속도 계산
    t1 = time.time()
    keypoints = surf.detect(image=gray, mask=None)
    t2 = time.time()
    print("SURF: %f sec\n" % (t2 - t1))