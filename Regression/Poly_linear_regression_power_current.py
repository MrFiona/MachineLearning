#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-27 11:19
# Author  : MrFiona
# File    : Poly_linear_regression_power_current.py
# Software: PyCharm Community Edition


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

path = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';', low_memory=False)

df.replace('?', np.nan, inplace=True)
df.dropna(how='any', inplace=True)

names = df.columns

X = df[names[2:4]]
Y = df[names[5]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#todo 多项式
model = Pipeline([
    ('StandardScaler', StandardScaler()), #todo 数据标准化
    ('PolynomialFeatures', PolynomialFeatures()), #todo 多项式
    ('LinearRegression', LinearRegression(fit_intercept=False)) #todo 线性回归
])

d_pool = np.arange(1, 5, 1)
t = np.arange(len(X_test))
color_combine = [('r', 'b'), ('c', 'm'), ('r', 'y'), ('b', 'g')]

for i,d in enumerate(d_pool):
    plt.subplot(4, 1, i+1)
    model.set_params(PolynomialFeatures__degree=d)
    model.fit(X_train, Y_train)
    y_predict = model.predict(X_test)
    score = model.score(X_test, Y_test)
    lin = model.get_params('LinearRegression')['LinearRegression']
    ss = model.get_params('StandardScaler')['StandardScaler']
    print(ss, '期望:', ss.mean_, ' 方差:', ss.var_)
    output = u'%d阶，系数为：' % d
    print(output, lin.coef_.ravel())
    print('决定系数R^2:\t', score)
    plt.plot(t,Y_test, color=color_combine[i][0], linestyle='-', label=u'真实值', lw=2, alpha=0.8)
    plt.plot(t, y_predict, color=color_combine[i][1], linestyle='-', label=u'%d阶:准确率=%.4f' % (d, score))
    plt.legend(loc='best')
    plt.grid()
    plt.ylabel(u'%d阶结果' % d, fontsize=20)

plt.suptitle(u'多项式线性回归预测电压和时间的关系', fontsize=20, color='r')
plt.show()
