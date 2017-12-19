#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('data.txt', sep='\t', header=None)
    estimator = KMeans(n_clusters=4)
    estimator.fit(data)
    label_pred = estimator.labels_ #获取聚类标签
    centroids = estimator.cluster_centers_ #获取聚类中心
    inertia = estimator.inertia_ # 获取聚类准则的总和

    colors = ['red', 'black', 'blue', 'green']
    marks = ['*','x','+','o']
    fig, ax = plt.subplots()
    for i, s in enumerate(label_pred):
        ax.scatter(data[0].loc[i], data[1].loc[i], c=colors[s], marker=marks[s],label=s, alpha=0.5, edgecolors='none')

    for i in range(len(centroids)):
        plt.annotate('center', xy=(centroids[i,0],centroids[i,1]), xytext=(centroids[i,0]+1,centroids[i,1]+1), arrowprops=dict(facecolor='red'))

    plt.show()