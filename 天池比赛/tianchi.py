#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2018-01-08 22:33
# Author  : MrFiona
# File    : tianchi.py
# Software: PyCharm


import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.svm import SVR, SVC, LinearSVC, LinearSVR, NuSVR
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve, auc
from sklearn.linear_model.coordinate_descent import ConvergenceWarning


start = time.time()

warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)
np.set_printoptions(linewidth=100, suppress=True)

data = pd.read_csv('d_train_20180102.csv', sep=',')
# data.drop(data.columns[19:23], axis=1, inplace=True)

columns_list = data.columns
print('columns_list:\t', columns_list)

#todo 将男女性别非数值型数据转化为0-1数值型数据
label_encoder = LabelEncoder()
data[columns_list[1]] = label_encoder.fit_transform(data[columns_list[1]].values)
# print data[columns_list[1]]

#todo 将时间字符串转化为时间数值型数据
def time_format(time_string):
    time_tuple = time.strptime(time_string, '%d/%m/%Y')
    return time.mktime(time_tuple)

data[columns_list[3]] = data[columns_list[3]].apply(lambda x: pd.Series(time_format(x)))
# print data[columns_list[3]]

# print(data)
for i in range(len(columns_list)):
    print('预处理之前columns_list[%d]:\t%s' %(i, columns_list[i]), sum(pd.isnull(data[columns_list[i]])))

# print data[columns_list[4]]
for column in range(len(columns_list[2:])):
    mean_value = data[columns_list[column+2]].median()
    print('column:\t%d' % column, mean_value)
    data[columns_list[column+2]].replace(np.nan, mean_value, inplace=True)

for i in range(len(columns_list)):
    print('预处理之后columns_list[%d]:\t%s' % (i, columns_list[i]), sum(pd.isnull(data[columns_list[i]])))
    print(data[columns_list[i]][data[columns_list[i]] < 0])

print(type(data),data.columns)

x = data[columns_list[1:-1]]
y = data[columns_list[-1]]

x = CCA(n_components=28).fit(x, y).transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,  random_state=0)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#todo 卡方选择
# ch2 = SelectKBest(chi2, k=10)
# x_train = ch2.fit_transform(x_train, y_train)
# x_test = ch2.transform(x_test)

#todo PCA降维
# pca = PCA(n_components=28)
# x_train = pca.fit_transform(x_train, y_train)
# x_test = pca.transform(x_test)

#todo 线性回归作预测
lr = LinearRegression(fit_intercept=True)
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)
# print(y_test.values, len(y_test.values), type(y_test.values))
# print(y_predict, len(y_predict), type(y_predict))
# fpr, tpr, thresholds = roc_curve(y_test.values, y_predict)
print('mean_absolute_error:\t', mean_absolute_error(y_test, y_predict))
print('mean_squared_error:\t', mean_squared_error(y_test, y_predict))
# print('roc_auc_score:\t', auc(fpr, tpr))
print('LinearRegression precision:\t', lr.score(x_test, y_test)*100)

#todo SVM回归作预测
# exhaustive_parameters = {'kernel':['rbf'], 'C':[1, 10, 100, 1000], 'gamma':np.logspace(-3, 1, 50)}
# exhaustive_parameters = {'fit_intercept':[True, False]}
# clf_SVC_exhaustive = GridSearchCV(estimator=LinearRegression(), param_grid=exhaustive_parameters)
# # clf_SVC_exhaustive = GridSearchCV(estimator=SVR(), param_grid=exhaustive_parameters)
# #todo 训练模型
# clf_SVC_exhaustive.fit(x_train, y_train)
# #todo 预测数据
# y_predict_cv = clf_SVC_exhaustive.predict(x_test)
# print(clf_SVC_exhaustive.score(x_test, y_test))
# print(clf_SVC_exhaustive.best_score_)
# print(clf_SVC_exhaustive.best_params_)
# print(clf_SVC_exhaustive.best_estimator_)
# print(clf_SVC_exhaustive.best_index_)

# #todo 通过GridSearchCV查找的SVR最优参数
# """
# C=1000, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
#   gamma=0.07543120063354615, kernel='rbf', max_iter=-1, shrinking=True,
#   tol=0.001, verbose=False
# """
# svm1 = SVR(C=1000, gamma=0.07543120063354615, kernel='rbf')
# svm1.fit(x_train, y_train)
# y_hat = svm1.predict(x_test)
# print('SVR precision:\t',svm1.score(x_test, y_test)*100)
#
# svm2 = LinearSVR(C=10)
# svm2.fit(x_train, y_train)
# y_hat5 = svm2.predict(x_test)
# print('SVC precision:\t',svm2.score(x_test, y_test)*100)
#
# #todo 决策树回归作预测
# decision_tree = DecisionTreeRegressor(criterion='mse')
# decision_tree.fit(x_train, y_train)
# y_hat1 = decision_tree.predict(x_test)
# print('Tree precision:\t',decision_tree.score(x_test, y_test)*100)
#
# #todo LassoCV回归作预测
# lc = LassoCV(alphas=np.logspace(-3, 1, 50), cv=10)
# lc.fit(x_train, y_train)
# y_hat2 = lc.predict(x_test)
# print('LassoCV precision:\t',lc.score(x_test, y_test)*100)
#
# #todo LassoCV回归作预测
# nb = NuSVR(C=1)
# nb.fit(x_train, y_train)
# y_hat6 = nb.predict(x_test)
# print('NuSVR precision:\t',nb.score(x_test, y_test)*100)
# print(cross_val_score(nb, x_test, y_test))


print('time total:\t', time.time() - start)


# print time.strptime('12/10/2017', '%d/%m/%Y')
# print time.mktime(time.strptime('12/10/2017', '%d/%m/%Y'))


"""
LinearRegression precision:	 22.6110003175
SVR precision:	 19.7354317235
LassoCV precision:	 22.6122650363
time total:	 0.4531097412109375
"""


#183749595205