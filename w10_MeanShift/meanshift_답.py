import numpy as np


def calc_euclidean_distance(p1, p2):  # 점과 점 사이 거리 측정하는 함수(유클리디안 거리)
    return np.sqrt(np.sum((p1 - p2) ** 2))


def calc_weight(x, kernel='flat'):  # 강의자료에서 k()에 해당하는 함수
    # 입력값이 1이하 작은 경우에만 가중치 1, 입력값이 1 초과인 경우 가중치 0으로 return
    # 그래서 1 초과인 데이터는 다음 중심점 계산에서 가중치가 0이 되어 참여하지 않음
    if x <= 1:  # x: 임의의 스칼라 값
        if kernel.lower() == 'flat':  # 모든 data 의 가중치 1
            return 1
        elif kernel.lower() == 'gaussian':  # 가까운 data 일수록 가중치가 더 높음
            return np.exp(-1 * (x ** 2))
        else:
            raise Exception("'%s' is invalid kernel" % kernel)
    else:
        return 0


def mean_shift(X, bandwidth, n_iteration=20, epsilon=0.001):
    # X: dataset, bandwidth: 탐색하는 반경, n_iteration: 최대 반복 횟수, epsilon: 수렴 판단 입실론 값
    centroids = np.zeros_like(X)

    for i in range(len(X)):
        centroid = X[i].copy()  # 초기 중심점(t_0) 설정 -> 각 datapoint 를 초기 중심점으로 할당
        prev = centroid.copy()

        t = 0

        while True:
            # 종료 조건 1- 반복 횟수가 n_iteration 을 초과하면 stop!
            if t > n_iteration:
                break

            # 현재 중심점으로부터 bandwidth 내에 있는 샘플들을 기반으로 새로운 군집 중심점 제안
            numerator = 0  # 분자
            denominator = 0  # 분모
            for sample in X:
                distance = calc_euclidean_distance(centroid, sample)
                weight = calc_weight(distance / bandwidth, 'flat')
                numerator += ((sample - centroid) * weight)
                denominator += weight

            if denominator == 0:
                shift = 0
            else:
                shift = numerator / denominator
            centroid += shift  # 다음 중심점 위치로 shift 만큼 이동

            # 종료 조건 2- 수렴했으면 stop!
            if calc_euclidean_distance(centroid, prev) < epsilon:
                break

            prev = centroid.copy()
            t += 1

        centroids[i] = centroid.copy()  # i 번재 sample 에 대한 최종 중심점

    return centroids


def mean_shift_with_history(X, bandwidth, n_iteration=20, epsilon=0.001):
    history = {}
    for i in range(len(X)):
        history[i] = []
    centroids = np.zeros_like(X)

    for i in range(len(X)):
        centroid = X[i].copy()  # 초기 중심점(t_0) 설정 -> 각 datapoint 를 초기 중심점으로 할당
        prev = centroid.copy()
        history[i].append(centroid.copy())

        t = 0
        while True:
            if t > n_iteration:  # t가 최대 반복 횟수를 초과한 경우
                break

            else:
                a = 0  # 분자
                b = 0  # 분모
                for j in range(len(X)):
                    a += X[j] * calc_weight(calc_euclidean_distance(np.array([0, 0]), ((X[j] - prev) / bandwidth)))
                    # Y_t: 시점 t 에서의 군집 중심점
                    b += calc_weight(calc_euclidean_distance(np.array([0, 0]), ((X[j] - prev) / bandwidth)))

                y_t1 = a / b  # y_(t+1)
                centroid = y_t1.copy()

                # if calc_euclidean_distance(np.array([0, 0]), centroid) <= epsilon:
                if calc_euclidean_distance(y_t1, prev) <= epsilon:
                    break

            prev = centroid.copy()
            t += 1

            history[i].append(centroid.copy())

        centroids[i] = centroid.copy()

    return centroids, history