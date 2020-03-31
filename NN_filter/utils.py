# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:44:42 2020

@author: adamw
"""
import matplotlib.pyplot as plt

def test_train_epoch_graph(history, train, test):
    plt.plot(history.history[train])
    plt.plot(history.history[test])
    plt.title('Model ', str(train))
    plt.ylabel(str(train))
    plt.xlabel('Epoch')
    plt.legend([str(train), str(test)], loc='upper right')
    plt.show()