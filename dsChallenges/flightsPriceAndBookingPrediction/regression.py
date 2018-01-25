#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 23:01:05 2017

@author: marcduda
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict   
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

df = pd.read_csv('features_regression.csv')
df = df.drop(['search_id'], axis=1) 
#df = df.values
y = df['fare_eur'].values
X = df.drop('fare_eur',1)
dummies = pd.get_dummies(X[['cabin_class']])
X = pd.concat([X, dummies], axis=1)
X = X.drop(['cabin_class','cabin_class_mixed','distance_code'], 1)
#%%
X_, X_test, y_, y_test = train_test_split(X,y, test_size=0.40, random_state=42)#[:20000] 
X_train, X_valid, y_train, y_valid = train_test_split(X_ ,y_ , test_size=0.40, random_state=42)


#%%
n_estimators = list(range(10,300,50))
metric = []
for n_estimator in n_estimators:
    forest = RandomForestRegressor(n_estimator, max_features='auto')
    forest.fit(X_train,y_train)
    scores = forest.predict(X_test)#cross_val_predict(forest, X_train, y_train, cv=10)
    metric.append(metrics.mean_squared_error(y_test, scores))
    print(metrics.mean_squared_error(y_test, scores))
    if n_estimator == 20:
        forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        
        for f in range(X_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()
print(metric)

#%%
import numpy as np
search = set(pd.read_csv('search_data.csv')['search_id'].unique())
booking = set(pd.read_csv('booking_data.csv')['booking_id'].unique())
all = search|booking
print(len(search))
print(len(booking))
print(len(all))

#%%
#encoder_cabin = LabelEncoder().fit(X['cabin_class'])
#X['cabin_class'] = encoder_cabin.transform(X['cabin_class'])
#encoder_distance = LabelEncoder().fit(X['distance_code'])
#X['distance_code'] = encoder_distance.transform(X['distance_code'])
#print(len(X['distance_code'].unique()))
#transform the categorical classes into an encoder
#encoded_y = np_utils.to_categorical(encoder_cabin)

print(pd.get_dummies(X[['cabin_class']]).head())
dummies = pd.get_dummies(X[['cabin_class']])
X = pd.concat([X, dummies], axis=1)
X = X.drop(['cabin_class','cabin_class_mixed'], 1)
#dummies_distance = pd.get_dummies(X[['distance_code']])
#X = pd.concat([X, dummies_distance], axis=1)
#X = X.drop(['distance_code'], 1)

#%%
best_n_estimator = 100
forest_model = RandomForestRegressor(n_estimator, max_features='auto')
forest_model.fit(X_train,y_train)
        
predicted_valid = forest_model.predict(X_valid)
print("Regression problem, metrics on validation set:")
print(metrics.classification_report(y_valid, predicted_valid))

