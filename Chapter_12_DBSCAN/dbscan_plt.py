#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'

import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('data.txt',sep='\t',header=None)
    print(data.head())
    y_pred = pd.read_csv('sub_class',header=None)
    print(y_pred.head())

    plt.scatter(data.iloc[:,0],data.iloc[:,1],c=y_pred)
    plt.show()