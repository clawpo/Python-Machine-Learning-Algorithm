#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'clawpo'

from fastFM import als
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 我们首先用pd.read_csv读入csv文件，切割前10000行数据，并区分出x与y
    train_file = pd.read_csv("data.txt", sep="\t", header=None)
    test_file = pd.read_csv('test_data.txt')
    images = train_file.iloc[:, :-1]
    labels = train_file.iloc[:, -1]
    # 分出80 % 的数据用于训练，20 % 的数据用于对训练效果进行评价
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,
                                                                            random_state=0)
    # 拟合模型
    model = als.FMRegression(n_iter=1000,init_stdev=0.1,rank=2,l2_reg_w=0.1,l2_reg_V=0.5)
    # print(train_images)
    # print(type(train_images))
    # print(type(train_labels))
    model.fit(train_images, train_labels)

    predict = model.predict(test_images)
    print(type(predict))
    print(type(test_labels))
    scroe = accuracy_score(predict, test_labels)
    print(scroe)
    # -------------------
    # data = pd.read_csv("data.txt", sep="\t", header=None)
    # data.columns = ["a", "b", "c"]
    # print(data.head())
    #
    # # print(data.values[:,:-1])
    # # print(data.values[:,-1])
    #
    # fm = als.FMRegression(n_iter=1000,init_stdev=0.1,rank=2,l2_reg_w=0.1,l2_reg_V=0.5)
    # fm.fit(data.values[:,:-1],data.values[:,-1])
    #
    # test_data = pd.read_csv("test_data.txt", sep="\t", header=None)
    # test_data.columns = ["a", "b", "c"]
    # print(test_data.head())
    #
    # y_pred = fm.predict(test_data.values[:,:-1])
    #
    # print(y_pred)
    # y_lab = pd.read_csv('predict_result')
    #
    # scroe = accuracy_score(y_lab, y_pred)
    # print(scroe)

    # print(data['c'].value_counts())
    # y = data['c'].value_counts().index
    # print(y)
    #
    # data1 = data.set_index(['c'])
    # colors = ['red', 'black', 'blue', 'green']
    # marks = ['*', 'x', '+', 'o']
    # fig, ax = plt.subplots()
    # for i, s in enumerate(y):
    #     ax.scatter(data1['a'].loc[s], data1['b'].loc[s], c=colors[i], marker=marks[i], label=s, alpha=0.5,
    #                edgecolors='none')
    # ax.legend(loc=5, fontsize=12)
    # plt.legend(loc='best')
    # # plt.ylim(-4.0, 16.0)
    # plt.show()
