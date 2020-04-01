# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:52:48 2020

@author: adamw
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from model import train_model

pd.set_option('display.max_colwidth', 1000)

dataframe = pd.read_csv("Data/sms_messages.csv")

df_label = dataframe['Category']
df_feature = dataframe['Message']

def create_doc_feature_df(sparse_mat, feature_names):
    return(pd.DataFrame.sparse.from_spmatrix(sparse_mat, columns=feature_names))
count_vect = CountVectorizer(stop_words='english')
count_vect.fit(df_feature)
feature = create_doc_feature_df(count_vect.transform(df_feature),count_vect.get_feature_names())
print(feature)

count_vect.fit(df_label)
label = create_doc_feature_df(count_vect.transform(df_label),count_vect.get_feature_names())
print(label)

x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)
train_model(x_train, x_test, y_train, y_test)