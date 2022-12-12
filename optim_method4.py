import cv2
import os
import numpy as np
import sys
from tqdm import tqdm

from generate_support_set import video_to_imgs, read_video
from method2 import imgs_to_video
from generate_support_set import save_imgs
from method4 import transform


def concat(imgs1, imgs2):
    assert len(imgs1) == len(imgs2)
    imgs = []
    for i in range(len(imgs1)):
        img1, img2 = imgs1[i], imgs2[i]
        img = np.concatenate([img1, img2], axis=1)
        imgs.append(img)
    return imgs


if __name__ == '__main__':
    '''
    本部分专门用于优化method4.py的过程，即优化从帧差图到二值化图的过程。
    通过分析每次算法运行的结果，思考原因，尝试新策略；再分析结果，思考原因尝试新策略。。。。。。这样的过程不断进行优化。
    此程序调用method4.py中的transform函数，我们需要在transform函数中进行修改，然后每次修改完运行此函数，得到结果进行分析。
    帧差法的缺点：1、运动物体内部灰度值相近，从而运动物理内部存在空洞；2、噪声多；等等
    '''
    print('------开始读入帧差视频------')
    diff_video = read_video('diff_videos/3.mp4')
    # 读帧差视频
    print('------将帧差视频转化为帧差图------')
    diff_imgs = video_to_imgs(diff_video)
    # 将帧差视频转化为帧差图
    print('------从帧差图生成二值化图------')
    bina_imgs = transform(diff_imgs)
    # 从帧差图生成二值化图
    print('------将帧差图与二值化图进行拼接------')
    imgs = concat(diff_imgs, bina_imgs)
    # 将帧差图与二值化图进行拼接
    print('------保存图片集------')
    save_imgs(imgs, 'optim_methods4_imgs')
    # 保存图片集
