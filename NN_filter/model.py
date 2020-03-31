# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:42:20 2020

@author: adamw
"""
from keras.models import Sequential
from keras.layers import Dense
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

encoder = args.encoder
hidden_neurons = args.neurons
num_epochs = args.num_epochs
learning_rate = args.lr

def train_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(50, input_dim=8440, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(x_train,y_train, epochs=20, batch_size=50, validation_data=(x_test, y_test))
    return history
