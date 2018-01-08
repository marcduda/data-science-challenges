#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:37:02 2017

@author: marcduda
"""
import pandas as pd
import numpy as np
train = pd.read_csv("train.csv")

train["Embarked"] = train["Embarked"].astype('category')
train.Embarked = (train.Embarked.cat.codes.replace(-1, np.nan)
                   .interpolate().astype(int).astype('category')
                   .cat.rename_categories(train.Embarked.cat.categories))

#print(train.groupby(['Sex','Pclass'])['Age'].mean())
train["Age"] = train.groupby(['Sex','Pclass','Embarked'])['Age'].transform(lambda x: x.fillna(x.mean()))


train1 = pd.read_csv("train.csv")
train1['Name'] = train1['Name'].astype(str)
train1['Ticket'] = train1['Ticket'].astype(str)
print(train1[train1['Ticket'].str.contains("113572")])#
#cabin from name ?!
print("%%%%%%%%%")
print(train1.loc[train1['Embarked'].isnull()])


trainQ = train.loc[train['Embarked'] == 'Q']
trainC = train.loc[train['Embarked'] == 'C']
trainS = train.loc[train['Embarked'] == 'S']

#%% 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np
#from keras.utils import np_utils
dummies_Sex = pd.get_dummies(train['Sex'])
dummies_Embarked = pd.get_dummies(train['Embarked'])
X = train.drop(['PassengerId','Survived','Name','Ticket','Cabin','Embarked','Sex'],1)
X = pd.concat([X, dummies_Sex,dummies_Embarked], axis=1)
y = train['Survived'].values
#%%
X, y = shuffle(X,y, random_state=0)
X_, X_test, y_, y_test = train_test_split(X,y, test_size=0.15, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_ ,y_ , test_size=0.15, random_state=42)
#print to verify if all classes are present in all the sets of data 

clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=10, tol=None)
clf.fit(X_train, y_train)  
predicted_SVM = clf.predict(X_test)
print("SVM part, metrics on test set:")
print(metrics.classification_report(y_test, predicted_SVM))

#%%
from sklearn.ensemble import RandomForestClassifier
RFclf = RandomForestClassifier(n_estimators=2000)
RFclf.fit(X, y)
predicted_RF = RFclf.predict(X_test)
print("RF part, metrics on test set:")
print(metrics.classification_report(y_test, predicted_RF))



test = pd.read_csv("test.csv")
test.count()
test["Age"] = test.groupby(['Sex','Pclass','Embarked'])['Age'].transform(lambda x: x.fillna(x.mean()))
test.count()
test['Fare']= test['Fare'].fillna(test['Fare'].mean())
test.count()
dummies_t_Sex = pd.get_dummies(test['Sex'])
dummies_t_Embarked = pd.get_dummies(test['Embarked'])
X_to_test = test.drop(['PassengerId','Name','Ticket','Cabin','Embarked','Sex'],1)
X_to_test = pd.concat([X_to_test, dummies_t_Sex,dummies_t_Embarked], axis=1)
y_to_test = RFclf.predict(X_to_test)

test['Survived'] = y_to_test
test[['PassengerId','Survived']].to_csv('prediction.csv',index=False)
