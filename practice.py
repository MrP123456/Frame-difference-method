import numpy as np

if __name__=='__main__':
    a=np.zeros([4,5])
    b=a.reshape([4*5])
    print(b.shape)
