# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:42:20 2020

@author: adamw
"""
#imports
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from utils import test_train_epoch_graph

#model function called by main.py
def train_model(x_train, x_test, y_train, y_test):
    words = x_test.shape[1] 
    model = Sequential()
    model.add(Dense(50, input_shape=(words,), activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    early_stop = EarlyStopping(patience=3)
    history = model.fit(x_train,y_train, epochs=20, batch_size=50, validation_data=(x_test, y_test), callbacks=[early_stop])
    predict = model.predict(x_test)
    print(predict)
    test_train_epoch_graph(history, 'loss', 'val_loss')
    test_train_epoch_graph(history, 'accuracy', 'val_accuracy')
    return history