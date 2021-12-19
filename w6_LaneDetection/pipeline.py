import numpy as np
import cv2


def set_region_of_interest(img, vertices):
    """
    :param img:       대상 이미지
    :param vertices:  이미지에서 남기고자 하는 영역의 꼭짓점 좌표 리스트
    :return:
    관심 영역만 마스킹 된 이미지
    """

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)

    return cv2.bitwise_and(img, mask)


def run(img):
    height, width = img.shape[:2]

    vertices = np.array([[(50, height),
                          (width // 2 - 45, height // 2 + 60),
                          (width // 2 + 45, height // 2 + 60),
                          (width - 50, height)]])

    # 123456 순서 중요 #
    # 1) BGR -> GRAY 영상으로 색 변환
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) 이미지 내 노이즈를 완화시키기 위해 blur 효과 적용
    blur_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    # cv2.GaussianBlur(src, Ksize, sigmaX)

    # 3) 캐니 엣지 검출을 사용하여 엣지 영상 검출
    edge_img = cv2.Canny(blur_img, 70, 175)

    # 4) 관심 영역(ROI; Region Of Interest)을 설정하여 배경 영역 제외
    ROI_img = set_region_of_interest(edge_img, vertices)

    # 5) 허프 변환을 사용하여 조건을 만족하는 직선 검출 (리턴: 찾은 선분의 시작점과 끝점 좌표 list)
    line_vertex = cv2.HoughLinesP(ROI_img, 1, np.pi / 180, 10, 10, 3)
    # cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap) → lines
    #   - image: 8bit, single-channel binary image, canny edge를 선 적용.
    #   - rho: r 값의 범위 (0 ~ 1 실수)
    #   - theta: 𝜃 값의 범위(0 ~ 180 정수)
    #   - threshold: 만나는 점의 기준, 숫자가 작으면 많은 선이 검출되지만 정확도가 떨어지고, 숫자가 크면 정확도가 올라감.
    #   - minLineLength: 선의 최소 길이. 이 값보다 작으면 reject.
    #   - maxLineGap – 선과 선사이의 최대 허용간격. 이 값보다 작으면 같은 선

    # 6) 찾은 직선들을 입력 이미지에 그리기
    result = np.copy(img)
    for line in line_vertex:
        for x1, y1, x2, y2 in line:
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 5)
            # cv2.line(img, pt1, pt2, color(BGR), thickness)

    return result