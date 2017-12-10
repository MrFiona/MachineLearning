# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from time import time
from sklearn.externals import joblib
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, stride):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# 输入图片，图片宽度，图片高度，图片通道数，分类数目
def deep_nn(x, width, height, channels, classes):
    x_image = tf.reshape(x, [-1, width, height, channels])

    # 卷积1：3*3*32
    conv1_width = 3
    conv1_height = 3
    conv1_channels = 32
    W_conv1 = weight_variable([conv1_width, conv1_height, channels, conv1_channels])
    b_conv1 = bias_variable([conv1_channels])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # Pooling1：2*2
    h_pool1 = max_pool(h_conv1, 2)

    # 卷积2：3*3*32
    conv2_width = 3
    conv2_height = 3
    conv2_channels = 64
    W_conv2 = weight_variable([conv2_width, conv2_height, conv1_channels, conv2_channels])
    b_conv2 = bias_variable([conv2_channels])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # Pooling2：2*2
    h_pool2 = max_pool(h_conv2, 2)

    # 卷积3：3*3*32
    conv3_width = 3
    conv3_height = 3
    conv3_channels = 64
    W_conv3 = weight_variable([conv3_width, conv3_height, conv2_channels, conv3_channels])
    b_conv3 = bias_variable([conv3_channels])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # Pooling3：2*2
    h_pool3 = max_pool(h_conv3, 2)

    # 卷积4：3*3*32
    conv4_width = 3
    conv4_height = 3
    conv4_channels = 64
    W_conv4 = weight_variable([conv4_width, conv4_height, conv3_channels, conv4_channels])
    b_conv4 = bias_variable([conv4_channels])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    # Pooling4：2*2
    h_pool4 = max_pool(h_conv4, 2)

    # 全连接1
    # print('h_pool4=', h_pool4)
    fc1_input = 16   # 此数据需要事先给定
    fc1_output = 1024
    W_fc1 = weight_variable([fc1_input * fc1_input * conv2_channels, fc1_output])
    b_fc1 = bias_variable([fc1_output])
    h_pool4_flat = tf.reshape(h_pool4, [-1, fc1_input * fc1_input * conv2_channels])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #全连接2
    W_fc2 = weight_variable([fc1_output, classes])
    b_fc2 = bias_variable([classes])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def next_batch(x, y, n, b):
    m = len(x)
    if b + n < m:
        return x[b:b+n], y[b:b+n], b+n
    return np.concatenate((x[b:], x[:n-m+b])), np.concatenate((y[b:], y[:n-m+b])), n-m+b


def build_model(x, y, x_test, y_test):
    print('开始时间：', datetime.now())
    t_start = time()

    classes = y.shape[1]
    width = height = int(np.sqrt(x.shape[1]/3)+0.5)
    x_ = tf.placeholder(tf.float32, [None, x.shape[1]])
    y_ = tf.placeholder(tf.float32, [None, classes])
    y_pred, dropout_prob = deep_nn(x_, width, height, 3, classes)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))
    train_step = tf.train.AdamOptimizer(3e-5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    b = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            xs, ys, b = next_batch(x, y, 300, b)
            train_step.run(feed_dict={x_: xs, y_: ys, dropout_prob: 0.5})
            if i % 2 == 0:
                train_accuracy = accuracy.eval(feed_dict={x_: xs, y_: ys, dropout_prob: 1.0})
                t_now = datetime.now()
                print('%d, %d:%d:%d, 训练集准确率：%g' % (i, t_now.hour, t_now.minute, t_now.second, train_accuracy))
        print('测试集准确率：%g' % accuracy.eval(feed_dict={x_: x_test, y_: y_test, dropout_prob: 1.0}))


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    width, height = 256, 256
    model_pkl = 'image.rf.pkl'
    path_data = u'.%sTraining' % os.sep
    if os.path.exists(model_pkl+'11'):
        print('正在载入模型...')
        model = joblib.load(model_pkl)
        print(path_data)
        kinds = os.listdir(path_data)
        names = []
        for kind in kinds:
            print(kind)
            names.append(kind)
    else:
        print('训练模型 - 读入训练数据')
        kinds = os.listdir(path_data)
        x = []
        y = []
        names = []
        for i, kind in enumerate(kinds):
            path_kind = u'%s%s%s' % (path_data, os.sep, kind)
            names.append(kind)
            print (kind)
            for num, t in enumerate(os.listdir(path_kind)):
                if num >= 1000:
                    print('单个类别图像足够多，忽略多余的图片')
                    break
                path_file = u'%s%s%s' % (path_kind, os.sep, t)
                # print path_file
                try:
                    image = Image.open(path_file)
                except:
                    print('Error')
                    continue
                image = image.resize((width, height), Image.ANTIALIAS)
                image = np.array(image)[:, :, :3]
                mirror_left_right = image[:, ::-1, :]
                x.append(image.reshape((-1, )))
                x.append(mirror_left_right.reshape((-1, )))
                y.append(i)
                y.append(i)
        x = np.array(x, dtype=float)
        lb = LabelBinarizer()
        y = lb.fit_transform(np.array(y))
        # x, x_test, y, y_test = train_test_split(x, y, train_size=0.75, random_state=0)
        N = y.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        train_num = int(N*0.75)
        print(N, train_num)
        build_model(x[:train_num], y[:train_num], x[train_num:], y[train_num:])
