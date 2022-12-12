import sys

import numpy as np
import cv2
import os
from tqdm import tqdm


def read_video(video_path):
    '''
    :return: 读入视频
    '''
    video = cv2.VideoCapture(video_path)
    return video


def video_to_imgs(video, to_gray=True):
    '''
    :param video: 视频
    :return: list[array] 视频转换后的图片列表
    '''
    imgs = []
    ret, frame = video.read()
    while ret:
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        imgs.append(frame)
        ret, frame = video.read()
    return imgs


def save_imgs(imgs, path):
    '''
    在path文件下保存保存imgs
    :param imgs: img组成的List
    :param path: 路径
    :return: None
    '''
    os.makedirs(path,exist_ok=True)
    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        img_path = os.path.join(path, str(i) + '.png')
        cv2.imwrite(img_path, img)


if __name__ == '__main__':
    video = read_video('videos/1.mp4')
    imgs = video_to_imgs(video)
    save_imgs(imgs, 'support_set')
