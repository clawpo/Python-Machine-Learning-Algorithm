#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    # 我们首先用pd.read_csv读入csv文件，切割前10000行数据，并区分出x与y
    train_file = pd.read_csv('../data/train.csv')
    test_file = pd.read_csv('../data/test.csv')
    images = train_file.iloc[0:10000,1:]
    labels = train_file.iloc[0:10000,0]
    # 分出80 % 的数据用于训练，20 % 的数据用于对训练效果进行评价
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
    # 拟合模型
    model = LogisticRegression().fit(train_images, train_labels)
    # 预测测试集中的数据
    predict = model.predict(test_images)
    # 根据预测结果对正确率进行打分
    scroe = accuracy_score(predict, test_labels)
    print(scroe)



