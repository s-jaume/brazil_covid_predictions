# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:51:34 2022

@author: sergio.jaume
"""

#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_excel("brazil_covid_dataset.xlsx")

#get %age of NaN
df_null = df.isnull().sum()

#drop the columns with all NAs
df.dropna(axis=1, how='all')
print(len(df.index))

#select columns, clean dataset
df = df.drop(["Patient ID"], axis=1)

fill=0
cols = df.select_dtypes(include='number').columns
df[cols] = df[cols].fillna(fill)

#enable on-hot encoding
df_dummy = pd.get_dummies(df, drop_first=(True))

#normalize the data
    #select columns with categorical variables
scaler = MinMaxScaler()
cols = df_dummy.select_dtypes(include='number').columns
    #normalize categorical variables
df_dummy[cols] = scaler.fit_transform(df_dummy[cols])

#seperate results from input data
Y = df_dummy[['SARS-Cov-2 exam result_positive']]
X = df_dummy.drop(["SARS-Cov-2 exam result_positive"], axis=1)

#split to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    train_size = 0.7,
                                                    stratify=Y)

#train lr model
lr = LogisticRegression()
lr.fit(X_train,Y_train)

#predict test data
Y_predict = lr.predict(X_test)
Y_prob = lr.predict_proba(X_test)

#accuracy and confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
score = lr.score(X_test, Y_test)

#90% accuracy!

# For future iterations,
# fill in numerical columns with the median of the column




