#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
if __name__ == '__main__':
    data = pd.read_csv('data.txt',sep='\t')
    p = plt.scatter(data.iloc[:,0],data.iloc[:,1], marker='x',color='r',s=30)
    plt.title('Linear Regression')
    plt.legend(loc='upper right')

    plt.plot()

    plt.show()