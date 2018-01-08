#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:00:03 2017

@author: marcduda
"""

import nltk.data 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPooling1D, Conv1D
from sklearn.preprocessing import LabelBinarizer
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_auc_score


X = np.load("datasetRawWithNoise.npy")
y = np.load("labelsRawWithNoise.npy")

X, y = shuffle(X,y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=39)
X_train = X_train/np.max(X_train)
X_test = X_test/np.max(X_test)

lb = LabelBinarizer()
y_encoded = lb.fit_transform(y_train)

y_binary_train = to_categorical(y_train)
y_binary_test = to_categorical(y_test)
#%% First architecture : fully connected layers
class_weight_dl = {0:2.5, 1:1}#65
dim_features = X_train.shape[1]
model_dl = Sequential()
model_dl.add(Dense(30, input_dim=dim_features, activation='softmax'))
model_dl.add(Dropout(.7))
model_dl.add(Dense(20, activation='relu'))
model_dl.add(Dense(20, activation='softmax'))
model_dl.add(Dropout(.7))
model_dl.add(Dense(10, activation='relu'))
model_dl.add(Dense(2, activation='sigmoid'))

# Compile model
model_dl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model_dl.fit(np.reshape(X_train,(-1,dim_features*1)), y_binary_train, epochs=20, batch_size=1000,class_weight = class_weight_dl, validation_split=0.1)

# evaluate the model
scores = model_dl.evaluate(np.reshape(X_test,(-1,dim_features*1)), y_binary_test)
print("\n%s: %.2f%%" % (model_dl.metrics_names[1], scores[1]*100))  #, y_test



#predict the labels of the test set and print some metrics to compare it with the correct labels
predictions = model_dl.predict(np.reshape(X_test,(-1,dim_features*1)))
prediction_binary = [i for i in np.argmax(predictions,1)]
print("DL part, metrics on test set:")
print(metrics.classification_report(y_test, prediction_binary))
print("DL part, roc on test set:")
print(roc_auc_score(y_test, prediction_binary))

#%% Second architecture : one-dimensional convolutional layers
class_weight_cnn = {0:3, 1:1}#65
model_cnn = Sequential()

model_cnn.add(Conv1D(filters=5, kernel_size=20,  padding='same', dilation_rate=1, activation='relu'
, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',input_shape=(dim_features, 1)))
#model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling1D(pool_size=5))
model_cnn.add(Dropout(.25))
model_cnn.add(Conv1D(filters=10, kernel_size=20, strides=1,padding='same', activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=5))
model_cnn.add(Dropout(.25))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dropout(.5))
model_cnn.add(Dense(2, activation='softmax'))


# Compile model
model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model_cnn.fit(X_train, y_binary_train, epochs=20, batch_size=1000,class_weight = class_weight_cnn, validation_split=0.1)#

# evaluate the model
scores_cnn = model_cnn.evaluate(X_test, y_binary_test, verbose=1)
print("\n%s: %.2f%%" % (model_cnn.metrics_names[1], scores_cnn[1]*100))  #, y_test


#predict the labels of the test set and print some metrics to compare it with the correct labels
predictions_cnn = model_cnn.predict(X_test)
prediction_binary_cnn = [i for i in np.argmax(predictions_cnn,1)]
print("CNN part, metrics on test set:")
print(metrics.classification_report(y_test, prediction_binary_cnn))
print("CNN part, roc on test set:")
print(roc_auc_score(y_test, prediction_binary_cnn))
