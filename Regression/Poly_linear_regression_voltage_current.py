#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-27 11:17
# Author  : MrFiona
# File    : Poly_linear_regression_voltage_current.py
# Software: PyCharm Community Edition


import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def time_format(x):
    t = time.strptime(' '.join(x), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

path = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';', low_memory=False)

names = df.columns

#todo 数据预处理
df.dropna(how='any', inplace=True)
X = df[names[0:2]]
Y = df[names[4]]
X = X.apply(lambda x: pd.Series(time_format(x)), axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#todo 多项式
model = Pipeline([
    ('StandardScaler', StandardScaler()), #todo 数据标准化
    ('PolynomialFeatures', PolynomialFeatures()), #todo 多项式
    ('LinearRegression', LinearRegression(fit_intercept=False)) #todo 线性回归
])

d_pool = np.arange(1, 5, 1)
t = np.arange(len(x_test))
color_combine = [('r', 'b'), ('c', 'm'), ('r', 'y'), ('b', 'g')]

for i,d in enumerate(d_pool):
    plt.subplot(4, 1, i+1)
    model.set_params(PolynomialFeatures__degree=d)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = model.score(x_test, y_test)
    lin = model.get_params('LinearRegression')['LinearRegression']
    output = u'%d阶，系数为：' % d
    print(output, lin.coef_.ravel())
    # print(output, lin.coef_)
    print('决定系数R^2:\t', score)
    plt.plot(t, y_test, color=color_combine[i][0], linestyle='-', label=u'真实值', lw=2, alpha=0.8)
    plt.plot(t, y_predict, color=color_combine[i][1], linestyle='-', label=u'%d阶:准确率=%.4f' % (d, score))
    plt.legend(loc='best')
    plt.grid()
    plt.ylabel(u'%d阶结果' % d, fontsize=20)

plt.suptitle(u'多项式线性回归预测电压和时间的关系', fontsize=20, color='r')
plt.show()
