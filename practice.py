import numpy as np

if __name__=='__main__':
    a=np.zeros([5,5])
    b=np.zeros([3,5])
    c=np.concatenate([a,b],0)
    d=np.ones([1,5])
    print(c.shape,d.shape)
    e=c-d
    print(e)