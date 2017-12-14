#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'
from sklearn.svm import SVC
from Chapter_4_SVM.svm_train import load_data_libsvm
from Chapter_4_SVM.svm_test import load_test_data

def load_result():
    data = []
    f = open('result')
    for line in f.readlines():
        lines = line.strip().split(' ')
        for i in range(0, len(lines)):
            data.append(lines[i])
    return data

if __name__ == '__main__':
    X,y = load_data_libsvm('heart_scale')
    # fit a SVM model to the data
    model = SVC()
    model.fit(X, y)
    print(model)
    # make predictions
    test_data = load_test_data('svm_test_data')
    predicted = model.predict(test_data)
    print(predicted)
