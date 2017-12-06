#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2017-12-06 22:42
# Author  : MrFiona
# File    : 手写数字识别.py
# Software: PyCharm Community Edition

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

## 加载数字图片数据
digits = datasets.load_digits()

## 获取样本数量，并将图片数据格式化
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

## 模型构建
classifier = svm.SVC(gamma=0.001)
## 使用二分之一的数据进行模型训练
classifier.fit(data[:int(n_samples / 2)], digits.target[:int(n_samples / 2)])

## 测试数据部分实际值和预测值获取
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

## 计算准确率
print(u"分类器%s的分类效果:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print(u"混淆矩阵为:\n%s" % metrics.confusion_matrix(expected, predicted))

## 进行图片展示
plt.figure(facecolor='gray', figsize=(12,5))
## 先画出5个预测失败的
images_and_predictions = list(zip(digits.images[n_samples / 2:][expected != predicted], expected[expected != predicted], predicted[expected != predicted]))
for index, (image,expection, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'预测值/实际值:%i/%i' % (prediction, expection))
## 再画出5个预测成功的
images_and_predictions = list(zip(digits.images[n_samples / 2:][expected == predicted], expected[expected == predicted], predicted[expected == predicted]))
for index, (image,expection, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 6)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'预测值/实际值:%i/%i' % (prediction, expection))

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()