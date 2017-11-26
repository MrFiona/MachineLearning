#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-26 21:41
# Author  : MrFiona
# File    : use_sklearn_preprocessing.py
# Software: PyCharm Community Edition


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline


mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#todo 用多项式线性回归做预测
#todo 加载数据
path = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';', low_memory=False)

#todo 数据预处理
df.replace('?', np.nan, inplace=True)
data = df.dropna(how='any')
names = df.columns
X = df[names[5:6]]
Y = df[names[4]]

models = [
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', LinearRegression(fit_intercept=False))
    ])
]
model = models[0]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

t = np.arange(len(x_test))
N = 5
d_pool = np.arange(1, N, 1)
m = d_pool.size
clors = []
for c in np.linspace(16711680, 255, m):
    c = c.astype(int)
    clors.append('#%06x' % c)

line_width = 3
plt.figure(figsize=(12, 6), facecolor='w')
for i, d in enumerate(d_pool):
    plt.subplot(N-1, 1, i+1)
    plt.plot(t, y_test, 'r-', label=u'真实值', ms=10)
    model.set_params(Poly__degree=d)
    model.fit(x_train, y_train)
    lin = model.get_params('Linear')['Linear']
    print model.get_params('Linear')
    output = u'%d阶, 系数为: ' % d
    print(output, lin.coef_.ravel())

    y_hat = model.predict(x_test)
    s = model.score(x_test, y_test)

    z = N - 1 if (d == 2) else 0
    label = u'%d阶, 准确率=%.3f' % (d,s)
    plt.plot(t, y_hat, color=clors[i], lw=line_width, alpha=0.75, label=label)
    plt.legend(loc='upper right')
    plt.grid()
    plt.ylabel(u'%d阶结果' % d, fontsize=12)

plt.legend(loc='lower right')
plt.suptitle(u'线性回归预测电压与电流之间的多项式关系', fontsize=20)
plt.grid()
plt.show()
