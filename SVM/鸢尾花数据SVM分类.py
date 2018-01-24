#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-12-04 16:13
# Author  : MrFiona
# File    : 鸢尾花数据SVM分类.py
# Software: PyCharm Community Edition


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


#todo 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
np.set_printoptions(linewidth=1000, suppress=True)

#todo 读取数据 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
#todo 数据文件路径
path = '../datas/iris.data'
data = pd.read_csv(path, header=None)
x, y = data.loc[:, 0:3], data.loc[:, 4]
y = pd.Categorical(y).codes
x = x[[0, 1]]
print(y)
#todo 数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=28, test_size=0.4)
"""
    svm.SVC API说明：
    功能：使用SVM分类器进行模型构建
    参数说明：
    C: 误差项的惩罚系数，默认为1.0；一般为大于0的一个数字，C越大表示在训练过程中对于总误差的关注度越高，也就是说当C越大的时候，对于训练集的表现会越好，
    但是有可能引发过度拟合的问题(overfitting)
    kernel：指定SVM内部函数的类型，可选值：linear、poly、rbf、sigmoid、precomputed(基本不用，有前提要求，要求特征属性数目和样本数目一样)；默认是rbf；
    degree：当使用多项式函数作为svm内部的函数的时候，给定多项式的项数，默认为3
    gamma：当SVM内部使用poly、rbf、sigmoid的时候，核函数的系数值，当默认值为auto的时候，实际系数为1/n_features
    coef0: 当核函数为poly或者sigmoid的时候，给定的独立系数，默认为0
    probability：是否启用概率估计，默认不启动，不太建议启动
    shrinking：是否开启收缩启发式计算，默认为True
    tol: 模型构建收敛参数，当模型的的误差变化率小于该值的时候，结束模型构建过程，默认值:1e-3
    cache_size：在模型构建过程中，缓存数据的最大内存大小，默认为空，单位MB
    class_weight：给定各个类别的权重，默认为空
    max_iter：最大迭代次数，默认-1表示不限制
    decision_function_shape: 决策函数，可选值：ovo和ovr，默认为None；推荐使用ovr；（1.7以上版本才有）
"""

#todo 数据SVM分类器构建
clf = svm.SVC(C=1.0, kernel='linear', decision_function_shape='ovr')
#todo 模型训练
clf.fit(x_train, y_train)
print(clf)
#todo 计算模型的准确度/精度
print(clf.score(x_train, y_train))
print(u'训练集准确率:', accuracy_score(y_train, clf.predict(x_train)))
print(clf.score(x_test, y_test))
print(u'测试集的准确率:', accuracy_score(y_test, clf.predict(x_test)))
#todo 计算决策函数的结构值以及预测值(decision_function计算的是样本x到各个分割平面的距离<也就是决策函数的值>)
print('decision_function:\n', clf.decision_function(x_train))
print(y_train)
print(clf.predict(x_train))
print(y_test)
print(clf.predict(x_test))

#todo 画图
N = 500
x1_min, x2_min = x.min()
x1_max, x2_max = x.max()

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
#todo 生成网格采样点

x1, x2 = np.meshgrid(t1, t2)
#todo 测试点
grid_show = np.dstack((x1.flat, x2.flat))[0]
#todo 预测分类值
grid_hat = clf.predict(grid_show)
#todo 使之与输入的形状相同
grid_hat = grid_hat.reshape(x1.shape)

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w')
#todo 绘制一个四边形网格, 背景颜色由cm_light指定
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
#todo 所有样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)
#todo 测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)
#todo label列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)
plt.show()

rf = RandomForestClassifier(200)

score = cross_val_score(rf, x, y, scoring='accuracy')
print np.mean(score)
print score