import numpy as np


def calc_euclidean_distance(p1, p2): # 점과 점 사이 거리 측정
    return np.sqrt(np.sum((p1 - p2) ** 2))


def calc_weight(x, kernel='flat'): # 강의자료에서 k()에 해당하는 함수
    if x <= 1: # x: 임의의 스칼라 값 
        if kernel.lower() == 'flat':
            return 1
        elif kernel.lower() == 'gaussian':
            return np.exp(-1 * (x ** 2))
        else:
            raise Exception("'%s' is invalid kernel" % kernel)
    else:
        return 0

    
def mean_shift(X, bandwidth, n_iteration=20, epsilon=0.001):  # X: dataset, bandwidth: 탐색에 사용하는 원의 직경, n_iteration: 최대 반복 횟수, epsilon: 수렴 판단 입실론 값
    centroids = np.zeros_like(X)

    for i in range(len(X)):        
        centroid = X[i].copy()  # 초기 중심점(t_0) 설정 -> 각 datapoint를 초기 중심점으로 할당
        prev = centroid.copy()
        
        t = 0

        while True:
            if (t > n_iteration): # t가 최대 반복 횟수를 초과한 경우
                break
            
            else:
                a = 0 
                b = 0 
                for j in range(len(X)):
                    a += X[j] * calc_weight(calc_euclidean_distance(np.array([0,0]),((X[j] - prev) / bandwidth))) # Y_t: 시점 t에서의 군집 중심점        
                    b += calc_weight(calc_euclidean_distance(np.array([0,0]),((X[j] - prev) / bandwidth)))
                
                y_t1 = a / b  # y_(t+1)
                centroid = y_t1.copy()
                
                if(calc_euclidean_distance(y_t1,prev) <= epsilon):
                    break
                    
            prev = centroid.copy()
            t += 1
        
        centroids[i] = centroid.copy()

    return centroids

    
def mean_shift_with_history(X, bandwidth, n_iteration=20, epsilon=0.001):
    history = {}
    for i in range(len(X)):
        history[i] = []
    centroids = np.zeros_like(X)   

    for i in range(len(X)):
        centroid = X[i].copy()  # 초기 중심점(t_0) 설정 -> 각 datapoint를 초기 중심점으로 할당
        prev = centroid.copy()
        history[i].append(centroid.copy())
        
        t = 0
        while True:
            if (t > n_iteration): # t가 최대 반복 횟수를 초과한 경우
                break
            
            else:
                a = 0 
                b = 0 
                for j in range(len(X)):
                    a += X[j] * calc_weight(calc_euclidean_distance(np.array([0,0]),((X[j] - prev) / bandwidth)))  # Y_t: 시점 t에서의 군집 중심점        
                    b += calc_weight(calc_euclidean_distance(np.array([0,0]),((X[j] - prev) / bandwidth)))
                
                y_t1 = a / b  # y_(t+1)
                centroid = y_t1.copy()
                
                if(calc_euclidean_distance(y_t1,prev) <= epsilon):
                    break

            prev = centroid.copy()
            t += 1

            history[i].append(centroid.copy())
        
        centroids[i] = centroid.copy()

    return centroids, history