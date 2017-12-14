#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'

from sklearn import datasets
import gzip
import numpy as np

from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split


def load_data():
    # 加载Iris数据集
    iris = datasets.load_iris()
    # 数据集中第3列和第4列数据表示花瓣的长度和宽度
    X = iris.data[:,[2,3]]
    y = iris.target
    # 类别已经转成了数字，0 = Iris - Setosa, 1 = Iris - Versicolor, 2 = Iris - Virginica.
    print(np.unique(y))
    print(Version(sklearn_version))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def cal_distance(x, y):
    return ((x - y) * (x - y).T)[0, 0]


def get_prediction(train_y, result):
    result_dict = {}
    for i in range(len(result)):
        if train_y[result[i]] not in result_dict:
            result_dict[train_y[result[i]]] = 1
        else:
            result_dict[train_y[result[i]]] += 1
    predict = sorted(result_dict.items(), key=lambda d: d[1])
    return predict[0][0]


def k_nn(train_data, train_y, test_data, k):
    # print test_data  
    m = np.shape(test_data)[0]  # 需要计算的样本的个数  
    m_train = np.shape(train_data)[0]
    predict = []

    for i in range(m):
        # 对每一个需要计算的样本计算其与所有的训练数据之间的距离  
        distance_dict = {}
        for i_train in range(m_train):
            distance_dict[i_train] = cal_distance(train_data[i_train, :], test_data[i, :])
            # 对距离进行排序，得到最终的前k个作为最终的预测  
        distance_result = sorted(distance_dict.items(), key=lambda d: d[1])
        # 取出前k个的结果作为最终的结果  
        result = []
        count = 0
        for x in distance_result:
            if count >= k:
                break
            result.append(x[0])
            count += 1
            # 得到预测  
        predict.append(get_prediction(train_y, result))
    return predict


def get_correct_rate(result, test_y):
    print('原始值：',np.mat(test_y))
    print('预测值：',np.mat(result))
    m = len(result)

    correct = 0.0
    for i in range(m):
        if result[i] == test_y[i]:
            correct += 1
    return correct / m


if __name__ == "__main__":
    # 1、导入  
    print("---------- 1、load data ------------")
    X_train, X_test, y_train, y_test = load_data()
    # 2、利用k_NN计算
    train_x = np.mat(X_train)
    test_x = np.mat(X_test)
    print("---------- 2、K-NN -------------")
    result = k_nn(train_x, y_train, X_test, 10)
    # 3、预测正确性
    print("---------- 3、correct rate -------------")
    print(get_correct_rate(result, y_test))