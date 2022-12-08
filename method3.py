import cv2
import numpy as np
import sys
from tqdm import tqdm

from generate_support_set import read_video, video_to_imgs
from method2 import imgs_to_video, calcu_diff
from generate_standard_support_set import load_imgs


def frame_difference_3(imgs, support_imgs):
    '''

    :param imgs: 输入图片
    :param support_imgs: 支持图片
    :return: 输出图片
    '''
    out_imgs = []
    m, h, w = len(support_imgs), imgs[0].shape[0], imgs[0].shape[1]
    for j in tqdm(range(len(imgs))):
        img = imgs[j]
        diff_imgs = None
        for i, support_img in enumerate(support_imgs):
            diff_img = calcu_diff(img, support_img, 1)

            if diff_imgs is None:
                diff_imgs = np.zeros([m, diff_img.shape[0], diff_img.shape[1]])
            diff_imgs[i] = diff_img
        # 此行可以根据结果修改
        # out_img = diff_imgs[0]
        sum_diff_imgs = diff_imgs.copy()
        sum_diff_imgs = sum_diff_imgs.reshape([sum_diff_imgs.shape[0], -1])
        sum_diff_imgs = np.sum(sum_diff_imgs, axis=1)
        index_min = np.argmin(sum_diff_imgs)
        out_img = diff_imgs[index_min]
        # out_img = np.min(diff_imgs, axis=0)

        out_img = out_img.astype('uint8')

        out_img = cv2.resize(out_img, (img.shape[1], img.shape[0]))
        # print(img1.shape,img2.shape,diff_img.shape)
        '''if j >= 5150:
            cat_img = np.concatenate([img, support_imgs[0], out_img], axis=1)
            cv2.imshow('1', cat_img)
            cv2.waitKey(0)'''

        out_imgs.append(out_img)
    return out_imgs


if __name__ == '__main__':
    print('------read_video------')
    video = read_video('videos/1.mp4')
    print('-------video_to_imgs------')
    imgs = video_to_imgs(video)
    print('------load_support_imgs------')
    support_imgs = load_imgs('standard_support_set')
    print('------frame_difference------')
    target_imgs = frame_difference_3(imgs, support_imgs)
    print('------imgs_to_video------')
    imgs_to_video(target_imgs, 'diff_videos/3.mp4')
    print('------over------')
