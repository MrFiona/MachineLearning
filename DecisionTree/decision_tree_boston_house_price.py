#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-12-03 00:43
# Author  : MrFiona
# File    : decision_tree_boston_house_price.py
# Software: PyCharm Community Edition

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


#todo 设置属性防止中文乱码
mpl.rcParams[u'font.sans-serif'] = [u'simHei']
mpl.rcParams[u'axes.unicode_minus'] = False

iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature_C = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

#todo 读取数据
path = '../datas/iris.data'
data = pd.read_csv(path, header=None)
# print(data, type(data))
#todo 获取X变量
x=data[list(range(4))]
# print(x, type(x))
#todo 把Y转换成分类型的0,1,2
y=pd.Categorical(data[4]).codes
# print(data[4])
print("总样本数目：%d;特征属性数目:%d" % x.shape)

#todo 数据分割（训练集和测试集）
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=14)
print("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))

depths = np.arange(1, 20)
err_list = []
for d in depths:
    clf = DecisionTreeRegressor(criterion='mse', max_depth=d)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    err = 1 - score
    err_list.append(err)
    print(u'%d深度, 正确率%.5f' % (d, score))

plt.figure(facecolor='w')
plt.plot(depths, err_list, 'ro-', lw=3)
plt.xlabel(u'决策树深度', fontsize=16)
plt.ylabel(u'错误率', fontsize=16)
plt.grid()
plt.title(u'决策树层次太多导致的拟合问题(过拟合和欠拟合)', fontsize=18)
plt.show()