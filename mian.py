import cv2
import numpy as np
import sys


def read_video():
    '''
    :return: 读入视频
    '''
    video_path = 'videos/1.mp4'
    video = cv2.VideoCapture(video_path)
    return video


def video_to_imgs(video):
    '''
    :param video: 视频
    :return: list[array] 视频转换后的图片列表
    '''
    imgs = []
    ret, frame = video.read()
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        imgs.append(frame)
        '''cv2.imshow('img', frame)
        cv2.waitKey(0)'''
        ret, frame = video.read()
    return imgs


def frame_difference(imgs):
    '''
    :param imgs: list[array] 图片列表
    :return: list[array] 帧差的图片列表
    '''
    diff_imgs = []
    for i in range(1, len(imgs)):
        img1, img2 = imgs[i - 1], imgs[i]
        # diff_img = abs(np.array(img2) - np.array(img1))
        diff_img = calcu_diff(img1, img2)
        diff_img = diff_img.astype('uint8')

        diff_img = cv2.resize(diff_img, (img1.shape[1], img1.shape[0]))
        # print(img1.shape,img2.shape,diff_img.shape)
        cat_img = np.concatenate([img1, img2, diff_img], axis=1)
        # cv2.imshow('1', cat_img)
        # cv2.waitKey(0)

        diff_imgs.append(diff_img)
    return diff_imgs


def calcu_diff(img1, img2):
    '''
    :param img1: [h,w]] 灰度图1
    :param img2: [h,w] 灰度图2
    :return: [h-border*2,w-border*2] 图1和图2的帧差
    '''
    # [368,640] [368.640]]
    border = 1
    # 扩大至周围几个像素范围内
    h, w = img1.shape[0], img1.shape[1]
    src_img = img2[border:h - border, border:w - border]
    src_img = src_img.reshape([1, src_img.shape[0], src_img.shape[1]])
    # print(src_img.shape)
    diff_img = None
    left, right, up, down = -border, border, -border, border
    for i in range(left, right + 1):
        for j in range(up, down + 1):
            tmp_img = img1[border + i:h - border + i, border + j:w - border + j]
            tmp_img = tmp_img.reshape([1, tmp_img.shape[0], tmp_img.shape[1]])
            if diff_img is None:
                diff_img = tmp_img
            else:
                diff_img = np.concatenate([diff_img, tmp_img], axis=0)
    des_img = np.abs(src_img - diff_img)
    '''
    做到这里
    '''
    des_img = np.min(des_img, axis=0)
    return des_img


def imgs_to_video(imgs):
    '''
    :param imgs: list[array] 图片列表
    :return: 转成视频
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 60
    size = (imgs[0].shape[0], imgs[0].shape[1])
    video = cv2.VideoWriter('diff_videos/2.mp4', fourcc, fps, size, 0)
    for img in imgs:
        # img=cv2.cvtColor(img)
        img = cv2.resize(img, size)
        video.write(img)
    video.release()


if __name__ == '__main__':
    print('------read_video------')
    video = read_video()
    print('-------video_to_imgs------')
    imgs = video_to_imgs(video)
    print('------frame_difference------')
    target_imgs = frame_difference(imgs)
    print('------imgs_to_video------')
    imgs_to_video(target_imgs)
    print('------over------')
