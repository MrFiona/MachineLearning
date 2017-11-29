#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-29 21:28
# Author  : MrFiona
# File    : poly_assign_task.py
# Software: PyCharm Community Edition


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


#todo 处理字符乱码
mpl.rcParams[u'font.sans-serif'] = [u'simHei']
mpl.rcParams[u'axes.unicode_minus'] = False

#todo 设置终端打印参数
np.set_printoptions(linewidth=1000, suppress=True)

#todo 加载数据
df = pd.read_csv('./datas/household_power_consumption_1000.txt',sep=';', low_memory=False)
names = df.columns
print names

#todo 预处理数据
df.replace('?', np.nan, inplace=True)
df.dropna(how='any')
X = df[names[2:4]]
Y = df[names[5]]

#todo 分割测试集合训练集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#todo 多项式线性回归
models = [
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', LinearRegression())
        ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', LassoCV(alphas=np.logspace(-3, 4, 50)))
    ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', RidgeCV(alphas=np.logspace(-3, 4, 50)))
        ]),
    Pipeline([
                ('ss', StandardScaler()),
                ('poly', PolynomialFeatures()),
                ('linear', ElasticNetCV(alphas=np.logspace(-3, 4, 50), l1_ratio=[.1, .5, .7, .9, .95, .99, 1]))
            ])
]

parameters = {
    "poly__degree": [3, 2, 1],
    "poly__interaction_only": [True, False],
    "poly__include_bias": [True, False],
    "linear__fit_intercept": [True, False]
}

title = ['LinearRegression', 'LassoCV', 'RidgeCV', 'ElasticNetCV']
colors = ['g', 'b', 'y', 'g']
fig = plt.figure(facecolor='w')
length = np.arange(len(x_test))

for i in range(4):
    plt.subplot(2, 2, i+1)
    #todo 网格搜索
    model = GridSearchCV(models[i], param_grid=parameters, n_jobs=1)
    #todo 训练模型
    model.fit(x_train, y_train)
    #todo 预测结果
    y_predict = model.predict(x_test)
    print(u'The %s optimal parameters:' % title[i], model.best_params_)
    print(u'The %s R^2:' % title[i], model.best_score_)
    plt.plot(length, y_test, 'r-', lw=2, label=u'电流真实值', alpha=0.75)
    plt.plot(length, y_predict, '%s-' % colors[i], lw=2, label=u'电流预测值', alpha=0.75)
    plt.title(u'The %s $R^2$: %.10f' % (title[i], model.best_score_))
    # todo 设置横纵坐标标签
    plt.xlabel(u'功率', fontsize=12, color='m')
    plt.ylabel(u'电流', fontsize=12, color='m')
    plt.legend(loc='best')

plt.grid()
plt.savefig('poly_test.png', format='png', dpi=fig.dpi)
plt.show()