#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


plt.style.use('ggplot')

if __name__ == '__main__':
    data = pd.read_csv('SoftInput.txt',sep='\t',header=None)
    data.columns=['a','b','c']
    print(data.head())

    model = LogisticRegression().fit(data.iloc[:,:-1],data.iloc[:,-1])

    predict = model.predict(data.iloc[:,:-1])




    scroe = accuracy_score(predict,data.iloc[:,-1])
    print(scroe)


