# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:52:48 2020

@author: adamw
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def create_doc_feature_df(sparse_mat, feature_names):
    return(pd.DataFrame.sparse.from_spmatrix(sparse_mat, columns=feature_names))

def convert_data(features, labels):
    count_vect = CountVectorizer()
    count_vect.fit(features)
    feature = create_doc_feature_df(count_vect.transform(features),count_vect.get_feature_names())
    print(feature)
    
    count_vect.fit(labels)
    label = create_doc_feature_df(count_vect.transform(labels),count_vect.get_feature_names())
    print(label)

def test_train_split(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test