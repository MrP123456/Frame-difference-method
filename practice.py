import numpy as np
import cv2

if __name__=='__main__':
    a=np.eye(100)
    k=np.ones([10,10])
    b=cv2.dilate(a,kernel=k)
    cv2.imshow('1',b)
    cv2.waitKey(0)

