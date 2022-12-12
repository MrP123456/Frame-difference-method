import cv2
import os
import numpy as np
import sys
from tqdm import tqdm

from generate_support_set import video_to_imgs, read_video
from method2 import imgs_to_video


def transform(imgs):
    threshold = 8
    iterations1 = 1
    iterations2 = 1

    trans_imgs = []
    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        trans_img = img.copy()

        trans_img = cv2.medianBlur(trans_img, ksize=3)
        # 中值滤波，用于平滑图像，用于去噪

        trans_img[trans_img < threshold] = 0
        trans_img[trans_img >= threshold] = 255
        # 二值化方法，threshold作为阈值，阈值作为影响很大的可调参数

        kernel = np.ones([3, 3], dtype=np.uint8)
        trans_img = cv2.morphologyEx(trans_img, cv2.MORPH_OPEN, kernel, iterations=iterations1)  # 开运算，即先腐蚀后膨胀，用于消除噪声
        trans_img = cv2.morphologyEx(trans_img, cv2.MORPH_CLOSE, kernel, iterations=iterations2)  # 闭运算，即先膨胀后腐蚀，用于消除小黑洞‘
        # iterations作为影响很大的可调参数

        trans_imgs.append(trans_img)
    return trans_imgs


if __name__ == '__main__':
    video = read_video('diff_videos/3.mp4')
    imgs = video_to_imgs(video)
    imgs = transform(imgs)
    imgs_to_video(imgs, 'diff_videos/4.mp4')
