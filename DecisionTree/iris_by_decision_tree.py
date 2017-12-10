#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-12-01 10:16
# Author  : MrFiona
# File    : iris_by_decision_tree.py
# Software: PyCharm Community Edition


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#todo 决策树
from sklearn import tree
#todo 分类树
from sklearn.tree import DecisionTreeClassifier
#todo 分割测试集和训练集
from sklearn.model_selection import train_test_split
#todo 管道
from sklearn.pipeline import Pipeline
#todo 数据归一化
from sklearn.preprocessing import MinMaxScaler
#todo 网格搜索交叉验证
from sklearn.model_selection import GridSearchCV
#todo 特征选择
from sklearn.feature_selection import SelectKBest
#todo 卡方统计量
from sklearn.feature_selection import chi2
#todo 主成分分析
from sklearn.decomposition import PCA


#todo 设置属性防止中文乱码
mpl.rcParams[u'font.sans-serif'] = [u'simHei']
mpl.rcParams[u'axes.unicode_minus'] = False

iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature_C = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

#todo 读取数据
path = './datas/iris.data'
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
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=14)
x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
print("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))

#todo 数据标准化
"""
    #StandardScaler (基于特征矩阵的列，将属性值转换至服从正态分布)
    #标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下
    #常用与基于正态分布的算法，比如回归
    #数据归一化
    #MinMaxScaler （区间缩放，基于最大最小值，讲数据转换到0,1区间上的）
    #提升模型收敛速度，提升模型精度
    #常见用于神经网络
    #Normalizer （基于矩阵的行，将样本向量转换为单位向量）
    #其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准
    #常见用于文本分类和聚类、logistic回归中也会使用，有效防止过拟合
"""
ss = MinMaxScaler()
#todo 用标准化方法将数据进行处理并转换
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
print("原始数据各个特征属性的调整最小值:",ss.min_) #todo min_ =  -min/(max-min)*1.0
print("原始数据各个特征属性的缩放数据值:",ss.scale_) #todo scale_ = 1.0/(max-min)

"""
    #特征选择：从已有的特征中选择出影响目标值最大的特征属性
    #常用方法：{ 分类：F统计量、卡方系数，互信息mutual_info_classif
    #{ 连续：皮尔逊相关系数 F统计量 互信息mutual_info_classif
    #SelectKBest（卡方系数）
"""
#todo 在当前的案例中，使用SelectKBest这个方法从4个原始的特征属性，选择出来3个，K默认为10
ch2 = SelectKBest(chi2, k=3)
#todo 训练并转换
x_train = ch2.fit_transform(x_train, y_train)
#todo 转换
x_test = ch2.transform(x_test)
select_name_index = ch2.get_support(indices=True)
print ("对类别判断影响最大的三个特征属性分布是:",ch2.get_support(indices=False))
print(select_name_index)

"""
    #降维：对于数据而言，如果特征属性比较多，在构建过程中，会比较复杂，这个时候考虑将多维（高维）映射到低维的数据
    #常用的方法：
    #PCA：主成分分析（无监督）
    #LDA：线性判别分析（有监督）类内方差最小，人脸识别，通常先做一次pca
"""
#todo 构建一个pca对象，设置最终维度是2维  注意：这里是为了后面画图方便，所以将数据维度设置了2维，一般用默认不设置参数就可以
pca = PCA(n_components=2)
#todo 训练并转换
x_train = pca.fit_transform(x_train)
#todo 转换
x_test = pca.transform(x_test)

#todo 决策树模型构建
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
#todo 模型训练
model.fit(x_train, y_train)
#todo 模型预测
y_hat = model.predict(x_test)
print(y_hat)

#todo 模型结果的评估
y_test2 = y_test.reshape(-1)
result = (y_test2 == y_hat)
# print(type(result), dir(result))
# print ("准确率:%.2f%%" % (np.mean(result) * 100))
#todo 准确率
print ("Score：", model.score(x_test, y_test))
print ("Classes:", model.classes_)

#todo 画图
#todo 横纵各采样多少个值
N = 100

#todo pca降维降到了2维
print(x_train.shape, x_test.shape)
#todo 分别求出每个特征列对应的最小最大值
x1_min = np.min((x_train.T[0].min(), x_test.T[0].min()))
x1_max = np.max((x_train.T[0].max(), x_test.T[0].max()))
x2_min = np.min((x_train.T[1].min(), x_test.T[1].min()))
x2_max = np.max((x_train.T[1].max(), x_test.T[1].max()))

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
print(t1.shape, t2.shape)
#todo 生成网格采样点
x1, x2 = np.meshgrid(t1, t2)
print(x2.shape, x1.shape)
#todo 测试点
x_show = np.dstack((x1.flat, x2.flat))[0]
# print(x_show, x_show.shape)

#todo 预测值
y_show_hat = model.predict(x_show)

#todo 使之与输入的形状相同
y_show_hat = y_show_hat.reshape(x1.shape[0], -1)
print(y_show_hat.shape)
print(y_show_hat[0])

#todo 画图
plt_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
plt_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_show_hat, cmap=plt_light)
#todo ravel()与flatten()功能相似，将数组拉值，成shape=(1,N)
plt.scatter(x_test.T[0], x_test.T[1], c=y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=plt_dark, marker='*')  # 测试数据
plt.scatter(x_train.T[0], x_train.T[1], c=y_train.ravel(), edgecolors='k', s=40, cmap=plt_dark)  # 全部数据
plt.xlabel(u'特征属性1', fontsize=15)
plt.ylabel(u'特征属性2', fontsize=15)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(True)
plt.title(u'鸢尾花数据的决策树分类', fontsize=18)
plt.show()