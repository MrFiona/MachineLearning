#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-28 15:22
# Author  : MrFiona
# File    : Poly_linear_regression_boston_house_predict.py
# Software: PyCharm Community Edition


import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler



names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

def notEmpty(s):
    return s != ''

mpl.rcParams[u'font.sans-serif'] = [u'simHei']
mpl.rcParams[u'axes.unicode_minus'] = False

warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)

np.set_printoptions(linewidth=100, suppress=True)

df = pd.read_csv('../datas/boston_housing.data', header=None)
# print(df.values)

data = np.empty((len(df), 14))
for i, d in enumerate(df.values):
    d = list(map(float,list(filter(notEmpty, d[0].split(' ')))))
    data[i] = d

x, y = np.split(data, (13,), axis=1)
y = y.ravel()

# print('x:\t', x, type(x))
# print('y:\t', y, type(y))
print ("样本数据量:%d, 特征个数：%d" % x.shape)
print ("target样本数据量:%d" % y.shape[0])

models = [
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', RidgeCV(alphas=np.logspace(-3, 1, 20)))
        ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', LassoCV(alphas=np.logspace(-3, 1, 20)))
        ])
]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

parameters = {
    "poly__degree": [3,2,1],
    "poly__interaction_only": [True, False],
    "poly__include_bias": [True, False],
    "linear__fit_intercept": [True, False]
}

titles = ['Ridge', 'Lasso']
colors = ['g-', 'b-']
plt.figure(figsize=(16, 8), facecolor='w')
ln_x_test = range(len(x_test))

plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'真实值')
for t in range(2):
    model = GridSearchCV(models[t], param_grid=parameters, n_jobs=1)
    model.fit(x_train, y_train)

    print("%s算法:最优参数:" % titles[t], model.best_params_)
    print("%s算法:R值=%.3f" % (titles[t], model.best_score_))
    y_predict = model.predict(x_test)
    plt.plot(ln_x_test, y_predict, colors[t], lw=t + 3, label=u'%s算法估计值,$R^2$=%.3f' % (titles[t], model.best_score_))

plt.legend(loc='upper left')
plt.grid(True)
plt.title(u"波士顿房屋价格预测")
plt.show()