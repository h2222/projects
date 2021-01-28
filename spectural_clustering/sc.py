import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from itertools import cycle, islice

np.random.seed(1)
# https://www.cnblogs.com/pinard/p/6221564.html
# https://blog.csdn.net/Jenny_oxaza/article/details/106850014?spm=1000.2123.3001.4430


def get2circles(n=1000):
    X,Y = datasets.make_circles(n, factor=0.5, noise=0.05)
    return X, Y

def euclid_distance(x1, x2, sqrt_flag=False):
    a = np.sum((x1-x2)**2)
    b = np.sqrt(a) if sqrt_flag else a
    return b

def euclid_distance_matrix(X):
    """
    计算L2相似度矩阵
    X shape: [3, 4]
    S shape: [3, 3]

    每行为向量
    1 2 3 4
    5 6 7 8
    9 10 11 12
    
    对角为相同向量, l2距离为0, 因为对称所以直接映射
    0 d1 d2 
    d1 0 d3
    d2 d3 0
    """
    X = np.array(X)
    size = len(X)
    S = np.zeros((size, size))
    for i in range(size):
        for j in range(i+1, size):
            S[i][j] = 1.0 * euclid_distance(X[i], X[j])
            S[j][i] = S[i][j]
    return S


def knn(S, k, sigma=1.0):
    """
    通过knn, 根据相似矩阵S得到邻接矩阵A 
    """
    # 初始化邻接矩阵
    size = len(S)
    adjacent_matrix = np.zeros((size, size))

    for i in range(size):
        dist_with_index = zip(S[i], range(size))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        # 选取当前点的k临近点
        neigber_id = [dist_with_index[m][1] for m in range(k+1)]
        # 建立临接矩阵(不是k近邻, aij为0, 是k近邻, aij为下面的计算)
        # a_ij = exp( - L2(xi - xj) / 2*simga^2 ) 
        for j in neigber_id:
            adjacent_matrix[i][j] = np.exp(-S[i][j]/(2*sigma**2))
            adjacent_matrix[j][i] = adjacent_matrix[i][j]
    return adjacent_matrix


def laplacian_matrix(adjacent_matrix):
    # 获取每个点对其他点的度的加和:[size, size] -> [size, ]
    degree_matrix = np.sum(adjacent_matrix, axis=1)
    # 拉普拉斯变换
    laplacian_matrix = np.diag(degree_matrix) - adjacent_matrix
    # normilze (D^(1/2) * L * D^(1/2))
    sqrt_degree_matrix = np.diag(1.0 / (degree_matrix**(0.5)))
    return np.dot(np.dot(sqrt_degree_matrix, laplacian_matrix), sqrt_degree_matrix)


"""
clustering
"""
def sp_kmean(laplacian_matrix):
    # 特征值和特征向量
    x, V = np.linalg.eig(laplacian_matrix)
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x:x[0])
    H = np.vstack([V[:, i] for (v, i) in x[:500]]).T
    sp_res = KMeans(n_clusters=2).fit(H)
    return sp_res

def org_kmean(data):
    org_res = KMeans(n_clusters=2).fit(data)
    return org_res


"""
ploting
"""
def plot(X, y_sp, y_km):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                          '#f781bf', '#a65628', '#984ea3',
                                          '#999999', '#e41a1c', '#dede00']),
                                        int(max(y_km) + 1))))
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_sp])
    plt.title("Spectral Clustering")
    plt.subplot(122)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_km])
    plt.title("Kmeans Clustering")
    plt.show()


if __name__ == "__main__":
    # 取数据
    vec, label = get2circles(n=500)
    # L2 相似矩阵
    S = euclid_distance_matrix(vec)
    # 根据knn找到邻矩阵
    A = knn(S, k=10)
    # 拉普拉斯矩阵
    L = laplacian_matrix(A)
    # 对比聚类
    sp_res = sp_kmean(L)
    org_res = org_kmean(vec)
    # 画图
    plot(vec, sp_res.labels_, org_res.labels_)