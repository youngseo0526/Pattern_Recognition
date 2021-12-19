import cv2

def pipline(img):
    # img = cv2.imread('./test_images/solidWhiteRight.jpg')  # .은 현재 경로
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0.0)
    # cv2.Canny(src, lowTH, highTH)
    edge_img = cv2.Canny(blurred_img, 70, 140)

    return edge_img


# 비디오 파일 불러오기
cap = cv2.VideoCapture('./test_videos/solidWhiteRight.mp4')

while True:
    ok, frame = cap.read()

    if not ok:
        break

    edge_img = pipline(frame)
    cv2.imshow('edge', edge_img)
    key = cv2.waitKey(0)  # ms 단위로 delay 됨 (1000ms = 1s)
    if key == ord('x'):  # x 누르면 종료
        break

cap.release()