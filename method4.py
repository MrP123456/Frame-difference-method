import cv2
import os
import numpy as np
import sys
from tqdm import tqdm

from generate_support_set import video_to_imgs, read_video
from method2 import imgs_to_video


def transform(imgs):
    threshold = 8
    iterations = 1

    trans_imgs = []
    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        trans_img = img.copy()

        kernel = np.ones([3, 3], dtype=np.uint8)
        # trans_img=cv2.morphologyEx(trans_img,cv2.MORPH_OPEN,kernel)
        # trans_img=cv2.morphologyEx(trans_img,cv2.MORPH_OPEN,kernel)
        # trans_img = cv2.erode(trans_img, kernel=kernel, iterations=iterations)
        # trans_img = cv2.dilate(trans_img, kernel=kernel, iterations=iterations)

        trans_img[trans_img < threshold] = 0
        trans_img[trans_img >= threshold] = 255
        trans_imgs.append(trans_img)
    return trans_imgs


if __name__ == '__main__':
    video = read_video('diff_videos/3.mp4')
    imgs = video_to_imgs(video)
    imgs = transform(imgs)
    imgs_to_video(imgs, 'diff_videos/4.mp4')
