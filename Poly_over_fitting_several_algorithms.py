#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-27 11:46
# Author  : MrFiona
# File    : Poly_over_fitting_several_algorithms.py
# Software: PyCharm Community Edition


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


mpl.rcParams[u'font.sans-serif'] = [u'simHei']
mpl.rcParams[u'axes.unicode_minus'] = False

np.random.seed(200)
np.set_printoptions(linewidth=1000, suppress=True)
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
x.shape = -1, 1
y.shape = -1, 1
# print(x)

models = [
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', LinearRegression(fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', ElasticNetCV(alphas=np.logspace(-3, 2, 50), l1_ratio=[.1, .5, .7, .9, .95, 1], fit_intercept=False))
    ])
]

plt.figure(facecolor='w')
degree = np.arange(1, N, 4)  # 阶
dm = degree.size
colors = []  # 颜色
for c in np.linspace(16711680, 255, dm):
    c = c.astype(int)
    colors.append('#%06x' % c)

model = models[0]
for i, d in enumerate(degree):
    plt.subplot(int(np.ceil(dm / 2.0)), 2, i + 1)
    plt.plot(x, y, 'ro', ms=10, zorder=N)

    model.set_params(Poly__degree=d)
    model.fit(x, y)

    lin = model.get_params('Linear')['Linear']
    output = u'%d阶，系数为：' % (d)
    print(output, lin.coef_.ravel())

    x_hat = np.linspace(x.min(), x.max(), num=100)
    x_hat.shape = -1, 1
    y_hat = model.predict(x_hat)
    s = model.score(x, y)

    z = N - 1 if (d == 2) else 0
    label = u'%d阶, 正确率=%.3f' % (d, s)
    plt.plot(x_hat, y_hat, color=colors[i], lw=2, alpha=0.75, label=label, zorder=z)

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)

plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.suptitle(u'线性回归过拟合显示', fontsize=22)
plt.show()



np.random.seed(200)
np.set_printoptions(linewidth=1000, suppress=True)
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
x.shape = -1, 1
y.shape = -1, 1

plt.figure(facecolor='w')
degree = np.arange(1, N, 2)  # 阶
dm = degree.size
colors = []  # 颜色
for c in np.linspace(16711680, 255, dm):
    c = c.astype(int)
    colors.append('#%06x' % c)
titles = [u'线性回归', u'Ridge回归', u'Lasso回归', u'ElasticNet']

for t in range(4):
    model = models[t]
    plt.subplot(2, 2, t + 1)
    plt.plot(x, y, 'ro', ms=10, zorder=N)

    for i, d in enumerate(degree):
        model.set_params(Poly__degree=d)

        model.fit(x, y.ravel())

        lin = model.get_params('Linear')['Linear']

        output = u'%s:%d阶，系数为：' % (titles[t], d)
        print(output, lin.coef_.ravel())

        x_hat = np.linspace(x.min(), x.max(), num=100)
        x_hat.shape = -1, 1

        y_hat = model.predict(x_hat)

        s = model.score(x, y)

        z = N - 1 if (d == 2) else 0
        label = u'%d阶, 正确率=%.3f' % (d, s)
        plt.plot(x_hat, y_hat, color=colors[i], lw=2, alpha=0.75, label=label, zorder=z)

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title(titles[t])
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.suptitle(u'各种不同线性回归过拟合显示', fontsize=22)
plt.show()