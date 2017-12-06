#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import linear_model
from Chapter_7_LinearRegression.linear_regression_train import load_data

if __name__ == '__main__':
    X, y = load_data("data.txt")

    model = linear_model.LinearRegression()
    model.fit(X, y)
    print(model.intercept_)  # 偏置b
    print(model.coef_)  #回归系数w

    # print(X.shape)
    # print(y.shape)
    #
    #
    # print(X)
    # print(y)

    # data = pd.read_csv('data.txt', sep='\t')
    data2=pd.DataFrame(X,columns=["a","b"])
    print(data2.head())
    data3 = pd.DataFrame(y,columns=["c"])
    p = plt.scatter(data2["b"], data3["c"], marker='x', color='r', s=30,)
    plt.title('Linear Regression')
    plt.legend(loc='best')
    plt.plot(X,model.predict(X), color='g',linewidth=2)

    plt.show()

    # data = pd.read_csv('data.txt', sep='\t')
    # p = plt.scatter(data.iloc[:,0], data.iloc[:,1], marker='x', color='r', s=30,)
    # plt.title('Linear Regression')
    # plt.legend(loc='best')
    # plt.plot(X,model.predict(X), color='g',linewidth=2)
    #
    # plt.show()