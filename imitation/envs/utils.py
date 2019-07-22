import numpy as np

def flat_to_one_hot(val, ndim):
    """

    >>> flat_to_one_hot(2, ndim=4)
    array([ 0.,  0.,  1.,  0.])
    >>> flat_to_one_hot(4, ndim=5)
    array([ 0.,  0.,  0.,  0.,  1.])
    >>> flat_to_one_hot(np.array([2, 4, 3]), ndim=5)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  1.,  0.]])
    """
    shape =np.array(val).shape
    v = np.zeros(shape + (ndim,))
    if len(shape) == 1:
        v[np.arange(shape[0]), val] = 1.0
    else:
        v[val] = 1.0
    return v

def one_hot_to_flat(val):
    """
    >>> one_hot_to_flat(np.array([0,0,0,0,1]))
    4
    >>> one_hot_to_flat(np.array([0,0,1,0]))
    2
    >>> one_hot_to_flat(np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0]]))
    array([2, 0, 1])
    """
    idxs = np.array(np.where(val == 1.0))[-1]
    if len(val.shape) == 1:
        return int(idxs)
    return idxs