# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:40:08 2016

@author: marc
"""


from __future__ import print_function
import numpy as np

# import the datasets and the corresponding labels
train_dataset = np.load("train_dataset.npy")
train_labels = np.load("train_label.npy")
valid_dataset = np.load("valid_dataset.npy")
valid_labels = np.load("valid_label.npy")
test_dataset = np.load("test_dataset.npy")

#%%
from keras.utils import np_utils
from keras.applications.inception_v3 import InceptionV3
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

nb_epoch = 10
nb_classes = 80
img_width, img_height = 140, 140
train_data_dir = "train/"
validation_data_dir = "val/"

# create the base pre-trained model
base_model = InceptionV3(include_top=False, weights='imagenet')

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer to match the 80 classes
predictions = Dense(nb_classes, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# encode the labels of the data
encoder = LabelEncoder()
encoder.fit(train_labels.ravel())
y_train = encoder.transform(train_labels.ravel())
y_train = np_utils.to_categorical(y_train, nb_classes)
encoder.fit(valid_labels.ravel())
y_valid = encoder.transform(valid_labels.ravel())
y_valid = np_utils.to_categorical(y_valid, nb_classes)

train_generator = train_datagen.flow(train_dataset, y_train)

validation_generator = test_datagen.flow(valid_dataset, y_valid)

print("start model")
history = model.fit_generator(
    train_generator, steps_per_epoch=len(train_dataset) / 200)

# compare the results when using or not a generator for the data
predictions_generator = model.predict_generator(validation_generator, 100)
predictions_generator = predictions_generator[:2729, :]
prediction_binary_generator = [i for i in np.argmax(predictions_generator, 1)]
prediction_label_generator = encoder.inverse_transform(
    prediction_binary_generator)
print("DL part, metrics on test set:")
print(metrics.classification_report(valid_labels, prediction_label_generator))


predictions = model.predict(valid_dataset)
prediction_binary = [i for i in np.argmax(predictions, 1)]
prediction_label = encoder.inverse_transform(prediction_binary)
print("DL part, metrics on test set:")
print(metrics.classification_report(valid_labels, prediction_label))


#%%
import os
predictions_test = model.predict(test_dataset)
prediction_binary_test = [i for i in np.argmax(predictions_test, 1)]
prediction_label_test = encoder.inverse_transform(prediction_binary_test)
#%%
import pandas as pd
image_test_files = os.listdir("test/")
test_label_df = pd.DataFrame(data=prediction_label_test, columns=['label'])
test_id_df = pd.DataFrame(data=image_test_files, columns=['id'])
data_test = pd.concat([test_id_df, test_label_df], axis=1)
data_test.to_csv('prediction_image.csv', index=False)
