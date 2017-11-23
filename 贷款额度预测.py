#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-23 20:18
# Author  : MrFiona
# File    : 贷款额度预测.py
# Software: PyCharm Community Edition


import pandas as pd
import numpy as np

path = r'C:\Users\Public\MachineLearning\train_u6lujuX_CVtuZ9i.csv'
df = pd.read_csv(path,header=0, sep=',')
# print df.info()

def num_missing(x):
    return sum(x.isnull())

print(df.apply(num_missing, axis=0))