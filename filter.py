# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: adamw
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pd.set_option('display.max_colwidth', 1000)

dataframe = pd.read_csv("Data/sms_messages.csv")

df_label = dataframe['Category']
df_features = dataframe.drop('Category', 1)
print(df_label.head(10))

X_train, X_test, y_train, y_test = train_test_split(dataframe.Message, dataframe.Category, test_size=0.5, random_state = 0)

text_pipe = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_pipe.fit(X_train, y_train);
print("Training score: ", text_pipe.score(X_train, y_train))
print("Testing score:  ", text_pipe.score(X_test, y_test))


X_test = [
    "Winner!",
    "Hate this app!",
    "Click here now",
    "This doesn't work super well",
]
print(text_pipe.predict(X_test))