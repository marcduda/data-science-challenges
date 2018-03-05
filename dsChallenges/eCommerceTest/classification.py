#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:55:26 2017

@author: marcduda
"""

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import pandas as pd

# load data
data_train = pd.read_csv("train.csv", encoding="ISO-8859-1")
data_train['description'] = data_train['description'].apply(
    lambda x: ' '.join(x.replace('\x85', '').split(' ')[:-3]))
print(data_train['description'][0])

# get the data to classify (only the description since the tilte is repeted in the description) and the corresponding label
X = pd.np.array(data_train['description'])
y = pd.np.array(data_train['category_id'])

# shuffle the data points and separate into training and testing points
X, y = shuffle(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                    random_state=42)
# print to verify if all classes are present in all the sets of data
print(set(y_train))
print(set(y_test))

# create the pipelin, first vectorization of the text and then extraction
# of the tf-idf statistics ; classification of the data using
# a linear svm classifier
class_weight = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
                10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1}

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None,
                                           class_weight=class_weight))])
text_clf.fit(X_train, y_train)
predicted_SVM = text_clf.predict(X_test)
print("SVM part, metrics on test set:")
print(metrics.classification_report(y_test, predicted_SVM))

from sklearn.model_selection import GridSearchCV
parameters = {'clf__max_iter': (5, 10, 15),
              'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# use the tuned classifier to predict the labels of the test.csv data
d_test = pd.read_csv("test.csv", encoding="ISO-8859-1")
d_test['description'] = d_test['description'].apply(
    lambda x: ' '.join(x.replace('\x85', '').split(' ')[:-3]))
d_test['predicted_category_id'] = text_clf.predict(d_test['description'])
# write the predictions in a file
results = d_test[['id', 'predicted_category_id']]
results.to_csv('predictions.csv', index=False)
