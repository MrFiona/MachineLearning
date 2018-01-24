#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-30 09:42
# Author  : MrFiona
# File    : boston_house_predict.py
# Software: PyCharm Community Edition


import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel


#todo 设置字符集，防止出现中文乱码
mpl.rcParams[u'font.sans-serif'] = [u'simHei']
mpl.rcParams[u'axes.unicode_minus'] = False

#todo 设置打印参数
np.set_printoptions(linewidth=1000, suppress=True)

#todo 捕捉警告
warnings.filterwarnings(action='ignore',category=ConvergenceWarning)

#todo 加载以及预处理数据
names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
"""
    CRIM：城镇人均犯罪率。
    ZN：住宅用地超过 25000 sq.ft. 的比例。
    INDUS：城镇非零售商用土地的比例。
    CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）。
    NOX：一氧化氮浓度。
    RM：住宅平均房间数。
    AGE：1940 年之前建成的自用房屋比例。
    DIS：到波士顿五个中心区域的加权距离。
    RAD：辐射性公路的接近指数。
    TAX：每 10000 美元的全值财产税率。
    PTRATIO：城镇师生比例。
    B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例。
    LSTAT：人口中地位低下者的比例。
    MEDV：自住房的平均房价，以千美元计。
"""
#todo 由于数据集没有表头所以header设置为None， 否则第一行会被默认为header
df = pd.read_csv('../datas/boston_housing.data', header=None)

data = np.empty((len(df), 14))
for index, value in enumerate(df.values):
    value_list = list(map(float, value[0].split()))
    data[index] = value_list

X, Y = np.split(data, (13,), axis=1)
# print X, type(X), len(X)
# print Y, type(Y), len(Y)

m = SelectFromModel(GradientBoostingRegressor()).fit_transform(X, Y.ravel())
print m, len(m)


#todo 转换格式
Y = Y.ravel()
print(data, len(data))

#todo 分割测试集和训练集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#todo 用Lasso回归算法做特征选择 L1正则可以降维
models = [
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures(degree=1, interaction_only=True, include_bias=True)),
        ('linear', LassoCV(alphas=np.logspace(-3, 1, 20), fit_intercept=False))
    ])
]

#todo 训练模型
model = models[0]
model.fit(x_train, y_train)
#todo 预测数据
y_predict = model.predict(x_test)
#todo R^2系数 准确率
score = model.score(x_test, y_test)
print(u'准确率:', score)
print(u'回归参数:', list(zip(names, model.get_params('linear')['linear'].coef_)))
print(u'回归系数:', model.get_params('linear')['linear'].intercept_)
print(u'均值:', list(zip(names, model.get_params('ss')['ss'].mean_)))
print(u'方差:', list(zip(names, model.get_params('ss')['ss'].var_)))



#todo 通过Lasso特征选择 CHAS回归参数为0，表明房价与改变量无关，去除该项数据重新做预测
# print(X, type(X), len(X))
#todo 去除无关项数据
X = np.delete(X, 3, axis=1)
#todo 重新分割测试集和训练集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
names = names.remove('CHAS')
print(names)

models = [
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression())
    ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', LassoCV(alphas=np.logspace(-3, 1, 20)))
        ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', RidgeCV(alphas=np.logspace(-3, 1, 20)))
        ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', ElasticNetCV(alphas=np.logspace(-3, 1, 20)))
        ])
]

parameters = {
    "poly__degree": [1,2,3],
    "poly__interaction_only": [True, False],
    "poly__include_bias": [True, False],
    "linear__fit_intercept": [True, False]
}

title = [u'LinearRegression', u'LassoCV', u'RidgeCV', u'ElasticNetCV']
color_combine = [('r', 'b'), ('c', 'm'), ('r', 'y'), ('b', 'g')]
#todo 画图
length = np.arange(len(x_test))
for num in range(4):
    model = GridSearchCV(models[num], param_grid=parameters, n_jobs=1)
    fig = plt.subplot(2, 2, num+1)
    fig.set_facecolor('w')
    fig.set_title(u'%s回归预测结果对比效果图' % title[num], fontdict={'fontsize': 15}, loc='center', color='r')
    #todo 训练模型
    model.fit(x_train, y_train)
    #todo 预测数据
    y_predict_cv = model.predict(x_test)
    print(u'%s准确率(R^2):' % title[num], model.score(x_test, y_test))
    print(u'%s最佳参数:' % title[num], model.best_params_)
    print(u'%s最佳score(R^2):' % title[num], model.best_score_)
    # print(u'%s回归参数:' % title[num], model.get_params())
    plt.plot(length, y_test, '%s-' % color_combine[num][0], label=u'真实值', lw=2, alpha=0.7, zorder=2)
    plt.plot(length, y_predict_cv, '%s-' % color_combine[num][1], label=u'预测值', lw=2, alpha=0.7)
    plt.xlabel(u'变量')
    plt.ylabel(u'房价')
    plt.legend([U'%s算法:R^2值=%.10f(%d 阶)' % (title[num], model.best_score_, model.best_params_['poly__degree'])], loc='upper left')
    plt.grid()

plt.savefig('boston_house_chart.png', format='png')
plt.show()


models = [
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', LinearRegression(fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', RidgeCV(alphas=np.logspace(-3, 1, 20), fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', LassoCV(alphas=np.logspace(-3, 1, 20), fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', ElasticNetCV(alphas=np.logspace(-3, 1, 20), l1_ratio=[.1, .5, .7, .9, .95, .99, 1], fit_intercept=False))
    ])
]

#todo 重新分割测试集和训练集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

titles = [u'LinearRegression回归', u'Ridge回归', u'Lasso回归', u'ElasticNet回归']
color = ['b', 'm', 'g']
for i in range(4):
    model = models[i]
    plt.subplot(2, 2, i + 1)

    plt.plot(length, y_test, 'r-', lw=2, alpha=0.7, zorder=2)
    for degree in range(0, 3):
        #todo 设置模型阶数
        model.set_params(Poly__degree=degree+1)
        model.fit(x_test, y_test)
        lin = model.get_params('Linear')['Linear']
        output = u'%s:%d阶，系数为：' % (titles[i], degree+1)
        print(output, lin.coef_.ravel())
        y_predict_cv = model.predict(x_test)
        s = model.score(X, Y)
        label = u'%d阶, 正确率=%.3f' % (degree+1, s)
        plt.plot(length, y_predict_cv, '%s-' % color[degree], label=label, lw=2, alpha=0.7)

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title(titles[i])
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)

plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.suptitle(u'各种不同线性回归过拟合显示', fontsize=22)
plt.savefig('boston_house_chart2.png', format='png')
plt.show()