import numpy as np
import sys


def k_means(features, num_clus, max_epoch=999999, min_distance=0.00001):
    '''
    :param features: 输入的特征 [n,l]。n：特征数量；l：特征维度
    :param num_clus: 聚类类别数量
    :param max_epoch: 最大迭代次数，终止条件
    :param min_distance: 最小误差，终止条件
    :return: [num_clus,l] 聚类中心
    '''
    mid_features = init_mid_features(features, num_clus)
    for epoch in range(max_epoch):
        labels_set = calcu_features_labels(features, mid_features)
        new_mid_features = calcu_mid_features(features, labels_set)
        distance = np.linalg.norm(mid_features - new_mid_features)
        if distance < min_distance:
            break
        mid_features = new_mid_features
    return mid_features


def init_mid_features(features, num_clus):
    n = len(features)
    random_index = np.random.choice(n, num_clus, replace=False)
    mid_features = features[random_index]
    return mid_features


def calcu_features_labels(features, mid_features):
    assert features.shape[1] == mid_features.shape[1]
    n, l = features.shape
    m = mid_features.shape[0]
    _featues, _mid_features = features.reshape([n, 1, l]), mid_features.reshape([1, m, l])
    dis = np.sum(np.power(_featues - _mid_features, 2), 2)
    labels = np.argmin(dis, axis=1)
    labels_set = dict()
    for i, label in enumerate(labels):
        if label not in labels_set:
            labels_set[label] = []
        labels_set[label].append(i)
    assert len(labels_set) == m
    labels_set = list(labels_set.values())
    assert sum([len(n) for n in labels_set]) == n
    return labels_set


def calcu_mid_features(features, labels_set):
    l, m = features.shape[1], len(labels_set)
    mid_features = np.zeros([m, l])
    for i, labels in enumerate(labels_set):
        sub_features = features[labels]
        mid_features[i] = np.mean(sub_features, axis=0)
    return mid_features


if __name__ == '__main__':
    a = np.random.randn(44, 235520)
    b = k_means(a, 10)
    print(b.shape)
