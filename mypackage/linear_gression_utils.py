import numpy as np

def correlation(x, y):
    pass

def normalize(arr):
    """"""
    min = np.min(arr)
    max = np.max(arr)
    range = max - min
    return (arr - min) / range

    >>> squarer = lambda t: t ** 2
>>> x = np.array([1, 2, 3, 4, 5])
>>> squarer(x)
array([ 1,  4,  9, 16, 25])