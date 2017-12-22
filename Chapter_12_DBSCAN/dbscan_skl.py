#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'

from sklearn import cluster,datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def test1():
    X1, y1 = datasets.make_circles(n_samples=5000, factor=.6,
                                   noise=.05)
    X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]],
                                 random_state=9)

    X = np.concatenate((X1, X2))
    # plt.scatter(X[:, 0], X[:, 1], marker='o')
    # plt.show()

    # y_pred = cluster.DBSCAN().fit_predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    # plt.show()

    # y_pred = cluster.DBSCAN(eps=0.1).fit_predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    # plt.show()

    y_pred = cluster.DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()

def test2():
    data = pd.read_csv('data.txt', sep='\t', header=None)
    dbscan = cluster.DBSCAN(eps=1.38, min_samples=5)
    y_pred = dbscan.fit_predict(data)
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1],c=y_pred)
    plt.show()


if __name__ == '__main__':
    test2()
