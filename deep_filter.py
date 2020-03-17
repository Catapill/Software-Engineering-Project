# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:53:15 2020

@author: adamw
"""
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv("Data/sms_messages.csv")
df_label = dataframe['Category']
df_feature = dataframe['Message']

my_tags = ['ham','spam']
plt.figure(figsize=(5,4))
df_label.value_counts().plot(kind='bar');