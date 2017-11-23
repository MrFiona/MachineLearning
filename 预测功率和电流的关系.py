#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-11-23 13:39
# Author  : MrFiona
# File    : 预测功率和电流的关系.py
# Software: PyCharm Community Edition


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False


path = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';', low_memory=False)

df.replace('?', np.nan, inplace=True)
data = df.dropna(how='any')

names = df.columns

X = df[names[2:4]]
Y = df[names[5]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, Y_train)
y_predict = lr.predict(X_test)

print(u'准确率:', lr.score(X_test, Y_test))

mse = np.average( (y_predict - Y_test)**2 )
rmse = np.sqrt(mse)

t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='upper right')
plt.title(u"线性回归预测功率和电流之间的关系", fontsize=20)
plt.xlabel(u'功率', fontsize=15)
plt.ylabel(u'电流', fontsize=15)
plt.grid(True)
plt.show()