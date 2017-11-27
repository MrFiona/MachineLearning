#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-27 15:57
# Author  : MrFiona
# File    : wine_quality_predict.py
# Software: PyCharm Community Edition


import pandas as pd


path1 = './datas/winequality-red.csv'
path2 = './datas/winequality-white.csv'

df1 = pd.read_csv(path1, sep=';')
df2 = pd.read_csv(path2, sep=';')

df = pd.concat([df1, df2], axis=0)


