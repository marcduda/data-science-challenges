#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:52:32 2017

@author: marcduda
"""

import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

nb_files_dict = {'CIV': 13087, 'COM': 2265,'CRIM':384,'SOC':12262}
list_of_files = glob.glob('./*.txt')
list_texts = []
list_labels = []
list_weights = []
#split the files into individual texts and get the associated label
for file_name in list_of_files:
    with open(file_name, 'rt', encoding='utf-8') as fp: #utf-8
        label = file_name.split(".")[1][1:]
        my_file = fp.read()
        texts = my_file.split("<<<<<<<<<<NEW>>>>>>>>>>")
        for text in texts:
            list_texts.append(text)
            list_labels.append(label)
            list_weights.append(1/nb_files_dict[label])
            
#divide the data with its labels into train, text and validation sets        
X, y = shuffle(list_texts,list_labels, random_state=0)
X_, X_test, y_, y_test = train_test_split(X,y, test_size=0.15, random_state=42)#[:20000] 

X_train, X_valid, y_train, y_valid = train_test_split(X_ ,y_ , test_size=0.15, random_state=42)
#print to verify if all classes are present in all the sets of data 
print(set(y_train))
print(set(y_test))
print(set(y_valid))

#%% SVM part: use tfidf features as input to a linear SVM classifier
class_weight = {'CIV': 1, 'COM': 10,'CRIM':10,'SOC':1}

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=10, tol=None,class_weight=class_weight))])
text_clf.fit(X_train, y_train)  
predicted_SVM = text_clf.predict(X_test)
print("SVM part, metrics on test set:")
print(metrics.classification_report(y_test, predicted_SVM))


#from sklearn.model_selection import GridSearchCV
#parameters = {'clf__max_iter':(5,10,15),
#              }
##'vect__ngram_range': [(1, 1), (1, 2)],
##              'tfidf__use_idf': (True, False),
##              'clf__alpha': (1e-2, 1e-3),
#gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
#gs_clf = gs_clf.fit(X_train, y_train)
#print(gs_clf.best_score_)                                 
#
#for param_name in sorted(parameters.keys()):
#    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

#%% SVM part: once the model is reworked and satisfying, predict the labels from the validation set to set how the model performs 
# on a data never seen and never used for remodelling.
predicted_valid_SVM = text_clf.predict(X_valid)
print("SVM part, metrics on validation set:")
print(metrics.classification_report(y_valid, predicted_valid_SVM))


#%% DL part: prepare features as input to cnn ie tokenize the texts

from nltk.corpus import stopwords # Import the stop word list
import nltk.data 
#print(stopwords.words("french") )
tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')

count_vect = CountVectorizer(ngram_range=(1, 1),stop_words = stopwords.words("french"), min_df = 10
                            ,) #
X_vectorizer = count_vect.fit_transform(X_train)
X_features = X_vectorizer
X_features_test = count_vect.transform(X_test).toarray()


#add weight since classes' sizes are heavily unbalanced
class_weight_dl = {0: 1, 1:10, 2:10, 3:1}

# encode class categories as integers to plot metrics after 
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)

#transform the categorical classes into an encoder
encoded_y = np_utils.to_categorical(encoded_Y)




#%% DL part: built the model and fit it to the training data

dim_features = X_features.shape[1]
model_dl = Sequential()
model_dl.add(Dense(20, input_dim=dim_features, activation='relu'))
model_dl.add(Dense(10, activation='relu'))
#model_dl.add(Dense(10, activation='relu'))
model_dl.add(Dense(4, activation='sigmoid'))

# Compile model
model_dl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model_dl.fit(X_features.toarray(), encoded_y, epochs=15, batch_size=200,class_weight = class_weight_dl)

# evaluate the model
scores = model_dl.evaluate(X_features.toarray(), encoded_y)
print("\n%s: %.2f%%" % (model_dl.metrics_names[1], scores[1]*100))  #, y_test

#predict the labels of the test set and print some metrics to compare it with the correct labels
predictions = model_dl.predict(X_features_test)
prediction_binary = [i for i in np.argmax(predictions,1)]
prediction_label = encoder.inverse_transform(prediction_binary)
print("DL part, metrics on test set:")
print(metrics.classification_report(y_test, prediction_label))


#%% DL part: once the model is reworked and satisfying, predict the labels from the validation set to set how the model performs 
# on a data never seen and never used for remodelling.
predictions_valid = model_dl.predict(count_vect.transform(X_valid).toarray())
prediction_binary_valid = [i for i in np.argmax(predictions_valid,1)]
prediction_label_valid = encoder.inverse_transform(prediction_binary_valid)
print("DL part, metrics on validation set:")
print(metrics.classification_report(y_valid, prediction_label_valid))

