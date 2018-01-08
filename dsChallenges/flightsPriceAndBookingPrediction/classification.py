#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:59:13 2017

@author: marcduda
"""

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv('features_classification.csv')
df = df.drop(['booking_id'], axis=1) 
#df = df.values
y = df['booked'].values
X = df.drop('booked',1)
dummies = pd.get_dummies(X[['cabin_class']])
X = pd.concat([X, dummies], axis=1)
X = X.drop(['cabin_class','cabin_class_mixed'], 1)

#%%
X_, X_test, y_, y_test = train_test_split(X,y, test_size=0.40, random_state=42)#[:20000] 
X_train, X_valid, y_train, y_valid = train_test_split(X_ ,y_ , test_size=0.40, random_state=42)


#%%
class_weight = {1: 30, 0: 1}

clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=10, tol=None,class_weight=class_weight)
clf.fit(X_train,y_train)
predicted_SVM = clf.predict(X_test)
print("SVM part, metrics on test set:")
print(metrics.classification_report(y_test, predicted_SVM))

#from sklearn.model_selection import GridSearchCV
#parameters = {'max_iter':(5,10,15),'alpha': (1e-2, 1e-3),'class_weight':({1: 10, 0: 1},{1: 20, 0: 1},{1: 30, 0: 1})}
#gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
#gs_clf = gs_clf.fit(X_train, y_train)
#print(gs_clf.best_score_)                                 
#
#for param_name in sorted(parameters.keys()):
#    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

#%%
predicted_valid_SVM = clf.predict(X_valid)
print("SVM part, metrics on validation set:")
print(metrics.classification_report(y_valid, predicted_valid_SVM))