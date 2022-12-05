import sys
import numpy as np
import os
import cv2

from generate_support_set import save_imgs
from my_kmeans import k_means


def load_imgs(path):
    imgs = []
    sub_paths = os.listdir(path)
    for sub_path in sub_paths:
        img_path = os.path.join(path, sub_path)
        img = cv2.imread(img_path, 0)
        imgs.append(img)
    return imgs


def img_clus(imgs, num_clus):
    n, h, w = len(imgs), imgs[0].shape[0], imgs[0].shape[1]
    features = np.zeros([n, h * w])
    for i, img in enumerate(imgs):
        features[i] = img.reshape([h * w])
    mid_features = k_means(features, num_clus=num_clus)
    imgs = []
    for mid_feature in mid_features:
        imgs.append(mid_feature.reshape([h, w]))
    return imgs


if __name__ == '__main__':
    imgs = load_imgs('support_set')
    standard_imgs = img_clus(imgs, num_clus=10)
    save_imgs(standard_imgs, 'standard_support_set')
