#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-26 00:08
# Author  : MrFiona
# File    : 预测电压和电流的关系.py
# Software: PyCharm Community Edition


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#todo 加载数据
path = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';', low_memory=False)

#todo 数据预处理
df.replace('?', np.nan, inplace=True)
data = df.dropna(how='any')
names = df.columns
X = df[names[5:6]]
Y = df[names[4]]

#todo 数据的标准化
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
# ss.fit(x_train)
# x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

#todo 建立、训练线性回归模型并进行预测
lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)
print(u'准确率:', lr.score(x_test, y_test))

#todo 作图
t = np.arange(len(x_test))
fig = plt.figure(facecolor='w')
plt.plot(t , y_test, lw=2, color='r', linestyle='-', label=u'电压真实值')
plt.plot(t , y_predict, lw=2, color='g', linestyle='-', label=u'电压预测值')
plt.legend(loc='best', shadow=True)
plt.xlabel(u'电流', fontsize=12, color='b')
plt.ylabel(u'电压', fontsize=12, color='b')
plt.title(u"线性回归预测电压和电流之间的关系", fontsize=20)
plt.grid()
plt.show()

