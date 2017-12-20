#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'

import pandas as pd
import matplotlib.pyplot as plt

def draw():
    data = pd.read_csv('data', sep='\t', header=None)
    labels = pd.read_csv('sub', sep='\t', header=None)
    labels_pred = labels.T

    colors = ['red', 'black', 'blue', 'green']
    marks = ['*','x','+','o']
    fig, ax = plt.subplots()
    for i in range(len(labels_pred)):
        ax.scatter(data[0].loc[i], data[1].loc[i], c=colors[labels_pred.iloc[i][0]],
                   marker=marks[labels_pred.iloc[i][0]],label=labels_pred.iloc[i][0], alpha=0.5, edgecolors='none')
    plt.show()


if __name__ == '__main__':
    draw()
