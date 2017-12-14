#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'


import pickle
import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import time


def load_data():
    """
    返回包含训练数据、验证数据、测试数据的元组的模式识别数据
    训练数据包含50，000张图片，测试数据和验证数据都只包含10,000张图片
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')
    f.close()
    return training_data, validation_data, test_data

if __name__ == '__main__':

    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    training_data, validation_data, test_data = load_data()
    # 传递训练模型的参数，这里用默认的参数
    clf = svm.SVC()
    # 进行模型训练
    clf.fit(training_data[0], training_data[1])
    # test
    # 测试集测试预测结果
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("%s of %s test values correct." % (num_correct, len(test_data[1])))
    print(accuracy_score(test_data[1],predictions))
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
