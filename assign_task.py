#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-29 20:47
# Author  : MrFiona
# File    : assign_task.py
# Software: PyCharm Community Edition


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


#todo 处理字符乱码
mpl.rcParams[u'font.sans-serif'] = [u'simHei']
mpl.rcParams[u'axes.unicode_minus'] = False

#todo 加载数据
df = pd.read_csv('./datas/household_power_consumption_1000.txt',sep=';')
names = df.columns
print names

#todo 预处理数据
df.replace('?', np.nan, inplace=True)
df.dropna(how='any')
X = df[names[2:4]]
Y = df[names[5]]

#todo 分割测试集合训练集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#todo 数据标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#todo 建立线性回归模型
lr = LinearRegression()
#todo 训练模型
lr.fit(x_train, y_train)
#todo 预测数据
y_predict = lr.predict(x_test)
print y_predict, type(y_predict)
print y_test, type(y_test)

#todo 画图显示结果
length = np.arange(len(x_test))
#todo 设定画布
fig = plt.figure(facecolor='w')
#todo 画两个折线趋势图
plt.plot(length, y_test, 'r-', label=u'电流真实值', alpha=0.8, lw=2)
plt.plot(length, y_predict, 'g-', label=u'电流预测值', alpha=0.8, lw=2)
#todo 设置横纵坐标标签
plt.xlabel(u'功率', fontsize=12, color='m')
plt.ylabel(u'电流', fontsize=12, color='m')
print(u'The correct rate of prediction:', lr.score(x_test, y_test))
#todo 设置图例位置为右上方
plt.legend(loc='upper right')
#todo 设置图表标题
plt.title(u'线性回归预测功率和电流之间的关系', fontsize=15, color='b')
#todo 开启网格
plt.grid(True)
#todo 显示图表
plt.savefig('test.png', format='png', dpi=fig.dpi)
plt.show()