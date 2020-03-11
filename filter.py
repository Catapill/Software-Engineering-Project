# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 1000)

dataframe = pd.read_csv("Data/sms_messages.csv")
dataframe.head(10)

df_label = dataframe['Category']
df_features = dataframe.drop('Category', 1)
print(df_label.head())
df_features.head()

text_pipe = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_pipe.fit(dataframe.Message, dataframe.Category);

X_test = [
    "Winner!",
    "Hi, are you okay?",
    "Make an appointment with me soon please",
    "Answer these qustions to get a prize",
]
print(text_pipe.predict(X_test))