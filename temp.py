import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("file.csv")
dataframe.head()

df_label = dataframe['Category']
df_features = dataframe.drop('Category', 1)
print(df_label.head())
df_features.head()

data = np.array(df_features)
label = np.array(df_label)
print(data.shape,df_label.shape)

print(dataframe.head())

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
x_train.shape

model = Sequential()
model.add(Dense(50, input_dim=8, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
