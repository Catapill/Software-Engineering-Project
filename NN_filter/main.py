# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:39:53 2020

@author: adamw
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import convert_data, test_train_split
from network import train_model
from graph import test_train_epoch_graph

pd.set_option('display.max_colwidth', 1000)

dataframe = pd.read_csv("Data/sms_messages.csv")
df_label = dataframe['Category']
df_feature = dataframe['Message']
