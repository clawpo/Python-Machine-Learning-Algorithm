#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'
import matplotlib.pyplot as plt
import pandas as pd

def draw():
    data = pd.read_csv('data.txt', sep='\t', header=None)
    labels = pd.read_csv('sub', sep='\t', header=None)
    labels_pred = labels[0]
    centroids = pd.read_csv('center', sep='\t', header=None)

    colors = ['red', 'black', 'blue', 'green']
    marks = ['*','x','+','o']
    fig, ax = plt.subplots()
    for i, s in enumerate(labels_pred):
        ax.scatter(data[0].loc[i], data[1].loc[i], c=colors[int(s)], marker=marks[int(s)],label=s, alpha=0.5, edgecolors='none')

    for i in range(len(centroids)):
        plt.annotate('center', xy=(centroids.iloc[i,0],centroids.iloc[i,1]), xytext=(centroids.iloc[i,0]+1,centroids.iloc[i,1]+1), arrowprops=dict(facecolor='red'))

    plt.show()

if __name__ == '__main__':
    draw()
