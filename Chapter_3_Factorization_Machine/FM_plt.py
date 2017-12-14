#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'

from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("data.txt", sep="\t", header=None)
    data.columns = ["a", "b", "c"]
    print(data.head())

    print(data['c'].value_counts())
    y = data['c'].value_counts().index
    print(y)

    data1 = data.set_index(['c'])
    colors = ['red', 'black', 'blue', 'green']
    marks = ['*', 'x', '+', 'o']
    fig, ax = plt.subplots()
    for i, s in enumerate(y):
        ax.scatter(data1['a'].loc[s], data1['b'].loc[s], c=colors[i], marker=marks[i], label=s, alpha=0.5,
                   edgecolors='none')
    ax.legend(loc=5, fontsize=12)
    plt.legend(loc='best')
    # plt.ylim(-4.0, 16.0)
    plt.show()