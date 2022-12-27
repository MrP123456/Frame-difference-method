import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm

from generate_support_set import read_video, video_to_imgs
from method2 import imgs_to_video


def def_args():
    parser = argparse.ArgumentParser(description='BS-GMM')
    parser.add_argument('--video_path', type=str, default='videos/1.mp4')
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--dev', type=float, default=2.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--init_omega', type=float, default=0.001)
    parser.add_argument('--init_sigma', type=float, default=15.)
    return parser.parse_args()


class GMM(object):
    def __init__(self, max_K):
        '''
        初始化GMM模型参数
        :param max_K: 最大混合模型数量
        '''
        self.max_K = max_K
        self.K = 0
        self.omega = []  # 权重
        self.mu = []  # 均值
        self.sigma = []  # 方差
        self.L = []  # 每个模型描述的是背景还是前景

    def sort_model(self):
        sort_list = []
        for k in range(self.K):
            weight = self.omega[k] / self.sigma[k]
            sort_list.append(weight)
        sorted_index = sorted(range(self.K), key=lambda x: sort_list[x], reverse=True)
        self.omega = list(np.array(self.omega)[sorted_index])
        self.mu = list(np.array(self.mu)[sorted_index])
        self.sigma = list(np.array(self.sigma)[sorted_index])
        self.L = list(np.array(self.L)[sorted_index])

    def update_omega(self):
        sum_omega = sum(self.omega)
        for k in range(self.K):
            self.omega[k] /= sum_omega

    def match(self, pixel):
        match_index = -1

        for k in range(self.K):
            if np.abs(pixel - self.mu[k]) / (np.sqrt(self.sigma[k])) <= args.dev:
                match_index = k
                break
        if match_index != -1:
            for k in range(self.K):
                M = 1. if k == match_index else 0.
                self.omega[k] = (1 - args.lr) * self.omega[k] + args.lr * M
            self.update_omega()
            rou = args.lr * gauss_prob(pixel, self.mu[match_index], self.sigma[match_index])
            rou = min(1, rou)
            self.mu[match_index] = (1 - rou) * self.mu[match_index] + rou * pixel
            self.sigma[match_index] = (1 - rou) * self.sigma[match_index] + rou * (pixel - self.mu[match_index]) ** 2
            self.sigma[match_index] = max(self.sigma[match_index], 1e-5)
            is_back = self.L[match_index]
        else:
            omega = args.init_omega
            L = True
            mu = pixel
            sigma = args.init_sigma
            if self.K < self.max_K:
                self.omega.append(omega)
                self.L.append(L)
                self.mu.append(mu)
                self.sigma.append(sigma)
                self.K += 1
            else:
                self.omega[-1] = omega
                self.L[-1] = L
                self.mu[-1] = mu
                self.sigma[-1] = sigma
            self.update_omega()
            is_back = False
        # self.sort_model()
        return is_back


def gauss_prob(x, mu, sigma):
    prob = 1 / np.sqrt(2 * np.pi * sigma + 1e-5) * np.exp(-1 / 2 * (x - mu) ** 2 / sigma)
    return prob


def main():
    video = read_video(args.video_path)
    imgs = video_to_imgs(video)

    '''for i, img in enumerate(imgs):
        imgs[i] = img[:10, :10]'''

    H, W = imgs[0].shape
    model = [[GMM(args.K)] * W for _ in range(H)]
    detect_imgs = []
    for img in tqdm(imgs):
        img = img.astype('float32')
        detect_img = np.zeros([H, W])
        for h in range(H):
            for w in range(W):
                pixel = img[h, w]
                is_back = model[h][w].match(pixel)
                detect_img[h][w] = 0 if is_back else 1
        detect_imgs.append(detect_img)
    imgs_to_video(imgs, 'method5.mp4')
    imgs_to_video(detect_imgs, 'method6.mp4')


if __name__ == '__main__':
    args = def_args()
    # main()
    video = read_video(args.video_path)
    imgs = video_to_imgs(video)
    GMM = cv2.createBackgroundSubtractorMOG2(
        history=999999,
        # 抛撒物保留的时长
        varThreshold = 16,
        # 阈值越低噪声越多
        detectShadows=True
        # 不知道干啥的
    )
    # GMM = cv2.createBackgroundSubtractorGMG()
    mask_imgs = []
    for img in imgs:
        mask_img = GMM.apply(img)
        cv2.imshow('mask', mask_img)
        cv2.waitKey(10)
        mask_imgs.append(mask_img)
    imgs_to_video(mask_imgs, 'method6.mp4')
