#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:00:03 2017

@author: marcduda
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPooling2D, Conv2D, MaxPooling1D, Conv1D
from sklearn.preprocessing import LabelBinarizer
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_auc_score

X = np.load("datasetSTFTWithNoise2D.npy")
y = np.load("labelsSTFTWithNoise2D.npy")
# data preparation, shuffling and splitting
X, y = shuffle(X, y, random_state=0)
X_train2D, X_test2D, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=39)
X_train2D = X_train2D/np.max(X_train2D)  # normalization of the training set
X_test2D = X_test2D/np.max(X_test2D)  # normalization of the testing set
(m, n, p) = X_train2D.shape
X_train = np.reshape(X_train2D, (m, n*p))

(i, j, k) = X_test2D.shape
X_test = np.reshape(X_test2D, (i, j*k))

# encoding the labels into categorical labels
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y_train)

y_binary_train = to_categorical(y_train)
y_binary_test = to_categorical(y_test)
#%% First architecture : fully connected layers
class_weight_dl = {0: 3.9, 1: 1}
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
model_dl.compile(loss='categorical_crossentropy',
                 optimizer='adam', metrics=['accuracy'])

# Fit the model
model_dl.fit(X_train, y_binary_train, epochs=20,
             batch_size=1000, class_weight=class_weight_dl)

# evaluate the model
scores = model_dl.evaluate(X_test, y_binary_test)
print("\n%s: %.2f%%" % (model_dl.metrics_names[1], scores[1]*100))


# predict the labels of the test set and print some metrics to compare it with the correct labels
predictions = model_dl.predict(X_test)
prediction_binary = [i for i in np.argmax(predictions, 1)]
print("STFT DL part, metrics on test set:")
print(metrics.classification_report(y_test, prediction_binary))
print("STFT DL part, roc on test set:")
print(roc_auc_score(y_test, prediction_binary))

#%% Second architecture : one-dimensional convolutional layers
class_weight_cnn = {0: 3, 1: 1}  # 65
model_cnn = Sequential()


model_cnn.add(Conv2D(filters=5, kernel_size=(20, 1), strides=5, padding='same', data_format='channels_last', dilation_rate=1,
                     activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape=(n, p, 1)))
model_cnn.add(MaxPooling2D(pool_size=(2, 1)))
model_cnn.add(Dropout(.25))
model_cnn.add(Conv2D(filters=10, kernel_size=(
    5, 1), strides=1, activation='relu'))
model_cnn.add(Activation('softmax'))
model_cnn.add(MaxPooling2D(pool_size=(2, 1)))
model_cnn.add(Dropout(.25))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dropout(.5))
model_cnn.add(Dense(2, activation='softmax'))


# Compile model
model_cnn.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

# Fit the model
model_cnn.fit(np.reshape(X_train2D, (m, n, p, 1)), y_binary_train,
              epochs=20, batch_size=1000, class_weight=class_weight_cnn)

# evaluate the model
scores_cnn = model_cnn.evaluate(np.reshape(
    X_test2D, (i, j, k, 1)), y_binary_test, verbose=1)
print("\n%s: %.2f%%" %
      (model_cnn.metrics_names[1], scores_cnn[1]*100))  # , y_test


# predict the labels of the test set and print some metrics to compare it with the correct labels
predictions_cnn = model_cnn.predict(np.reshape(X_test2D, (i, j, k, 1)))
prediction_binary_cnn = [i for i in np.argmax(predictions_cnn, 1)]
print("STFT 2D CNN part, metrics on test set:")
print(metrics.classification_report(y_test, prediction_binary_cnn))
print("STFT 2D CNN part, roc on test set:")
print(roc_auc_score(y_test, prediction_binary_cnn))


#%% Third architecture : two-dimensional convolutional layers
class_weight_cnn1D = {0: 3, 1: 1}  # 65
model_cnn1D = Sequential()

model_cnn1D.add(Conv1D(filters=5, kernel_size=20,  padding='same', dilation_rate=1, activation='relu',
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape=(dim_features, 1)))
model_cnn1D.add(MaxPooling1D(pool_size=5))
model_cnn1D.add(Dropout(.25))
model_cnn1D.add(Conv1D(filters=10, kernel_size=20, strides=1,
                       padding='same', activation='relu'))
model_cnn1D.add(MaxPooling1D(pool_size=5))
model_cnn1D.add(Dropout(.25))
model_cnn1D.add(Flatten())
model_cnn1D.add(Dense(50, activation='relu'))
model_cnn1D.add(Dropout(.5))
model_cnn1D.add(Dense(2, activation='softmax'))


# Compile model
model_cnn1D.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

# Fit the model
model_cnn1D.fit(np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)), y_binary_train,
                epochs=20, batch_size=1000, class_weight=class_weight_cnn, validation_split=0.1)

# evaluate the model
scores_cnn1D = model_cnn1D.evaluate(np.reshape(
    X_test, (X_test.shape[0], X_test.shape[1], 1)), y_binary_test, verbose=1)
print("\n%s: %.2f%%" %
      (model_cnn.metrics_names[1], scores_cnn[1]*100))  # , y_test

# predict the labels of the test set and print some metrics to compare it with the correct labels
predictions_cnn1D = model_cnn1D.predict(np.reshape(
    X_test, (X_test.shape[0], X_test.shape[1], 1)))
prediction_binary_cnn1D = [i for i in np.argmax(predictions_cnn1D, 1)]
print("STFT 1D CNN part, metrics on test set:")
print(metrics.classification_report(y_test, prediction_binary_cnn1D))
print("STFT 1D CNN part, roc on test set:")
print(roc_auc_score(y_test, prediction_binary_cnn1D))
