import cv2
import os
import numpy as np
import sys
from tqdm import tqdm

from generate_support_set import video_to_imgs, read_video
from method2 import imgs_to_video


def add_contours(imgs, src_imgs):
    assert len(imgs) == len(src_imgs)
    des_imgs = []
    for i in tqdm(range(len(imgs))):
        img, src_img = imgs[i], src_imgs[i].copy()
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 16:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if y < 50 and x > 480:
                continue
            cv2.rectangle(src_img, [x, y], [x + w, y + h], [0, 255, 0], 2)
        des_imgs.append(src_img)
    return des_imgs


def imgs_concat(imgs1, imgs2):
    imgs = []
    assert len(imgs1) == len(imgs2)
    for i in tqdm(range(len(imgs1))):
        img1, img2 = imgs1[i], imgs2[i]
        if len(img1.shape) == 2:
            img1 = img1.reshape([img1.shape[0], img1.shape[1], 1]).repeat(3, axis=2)
        if len(img2.shape) == 2:
            img2 = img2.reshape([img2.shape[0], img2.shape[1], 1]).repeat(3, axis=2)
        img = np.concatenate([img1, img2], axis=1)
        imgs.append(img)
    return imgs


if __name__ == '__main__':
    video = read_video('diff_videos/4.mp4')
    gray_imgs = video_to_imgs(video)
    src_video = read_video('videos/1.mp4')
    src_imgs = video_to_imgs(src_video, to_gray=False)
    des_imgs = add_contours(gray_imgs, src_imgs)
    imgs_to_video(des_imgs, 'diff_videos/5.mp4', is_gray=False)
