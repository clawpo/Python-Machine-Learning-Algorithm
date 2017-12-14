#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'

from sklearn import datasets
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 加载Iris数据集
    iris = datasets.load_iris()
    # 类别已经转成了数字，0 = Iris - Setosa, 1 = Iris - Versicolor, 2 = Iris - Virginica.
    y = iris.target

    colors = ['red', 'black', 'blue', 'green']
    marks = ['*','x','+','o']
    fig, ax = plt.subplots()
    for i, s in enumerate(y):
        # 数据集中第3列和第4列数据表示花瓣的长度和宽度
        ax.scatter(iris.data[:, [2]][i], iris.data[:, [3]][i], c=colors[s], marker=marks[s],label=s, alpha=0.5, edgecolors='none')
    # ax.legend(loc=5, fontsize=12)
    # plt.legend(loc='best')
    plt.ylim(0.0,3.0)
    plt.xlim(0.0,8.0)
    plt.show()