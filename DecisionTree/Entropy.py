# coding:utf-8

import numpy as np

if __name__ == '__main__':
    a = np.array([36, 32, 37], dtype=float)
    a /= a.sum()
    print a
    print np.log(a)
    print a * np.log(a)
    print -np.sum(a * np.log2(a))