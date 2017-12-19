#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('data.txt', sep='\t', header=None)
    num_clusters = 4
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=300,
                        n_init=40, init='k-means++', n_jobs=-1)
    km_cluster.fit(data)
    label_pred = km_cluster.labels_ #获取聚类标签
    centroids = km_cluster.cluster_centers_ #获取聚类中心


    colors = ['red', 'black', 'blue', 'green']
    marks = ['*','x','+','o']
    fig, ax = plt.subplots()
    for i, s in enumerate(label_pred):
        ax.scatter(data[0].loc[i], data[1].loc[i], c=colors[s], marker=marks[s],label=s, alpha=0.5, edgecolors='none')

    for i in range(len(centroids)):
        plt.annotate('center', xy=(centroids[i,0],centroids[i,1]), xytext=(centroids[i,0]+1,centroids[i,1]+1), arrowprops=dict(facecolor='red'))

    plt.show()