# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:51:42 2020

@author: adamw
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import keras
import numpy as np
from keras.datasets import reuters

pd.set_option('display.max_colwidth', 1000)

dataframe = pd.read_csv("Data/sms_messages.csv")

df_label = dataframe['Category']
df_feature = dataframe['Message']

def create_doc_feature_df(sparse_mat, feature_names):
    return(pd.DataFrame.sparse.from_spmatrix(sparse_mat, columns=feature_names))
# -----
count_vect = CountVectorizer(stop_words='english')
count_vect.fit(df_feature)
feature = create_doc_feature_df(count_vect.transform(df_feature),count_vect.get_feature_names())
print(feature)

count_vect.fit(df_label)
label = create_doc_feature_df(count_vect.transform(df_label),count_vect.get_feature_names())
print(label)

x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)
x_train.shape

model = Sequential()
model.add(Dense(50, input_dim=8440, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train,y_train, epochs=200, batch_size=32, validation_data=(x_test, y_test))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()