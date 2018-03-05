# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:30:14 2017

@author: marc
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('newsletter_sent.csv',
                    sep=';', header=0, na_values=-1)
test = pd.read_csv('recipients.csv',
                   sep=';', header=0, na_values=-1)
test['open'] = np.nan
df = pd.concat([train, test], axis=0)
print(df.count())
print(test.count())
print(train.count())
# %% The train and test sets have missing values so we will fill those using
# jointly the test and train set for a better accuracy. The missing values
# will be replaced with the mean of the complete values grouped by open, email,
# gender.
print(df.describe())
dfgroup = df.groupby(['email', 'gender'])
df['open_rate'] = dfgroup['open_rate'].transform(lambda x: x.fillna(round(x.mean())))
df['geo'] = dfgroup['geo'].transform(lambda x: x.fillna(round(x.mean())))
df['device'] = dfgroup['device'].transform(lambda x: x.fillna(round(x.mean())))
print(df.count())

# %% Since there are a lot of categorical data, we use a tree-based model to
# predict if the email will be opened or not. The scikit-learn implementation
# of such algorithms doesn't deal with catégorical data that is represented
# by an integer or a float so we have to first encode those values:
for col in ['email', 'gender', 'geo', 'device']:
    df[col] = df[col].astype('category')

dum_email = pd.get_dummies(df[['email']])
dum_gender = pd.get_dummies(df[['gender']])
dum_geo = pd.get_dummies(df[['geo']])
dum_device = pd.get_dummies(df[['device']])
df = pd.concat([df, dum_email, dum_gender, dum_geo, dum_device], axis=1)
df = df.drop(['email', 'gender', 'geo', 'device'], 1)

# %% A simple tree doesn't performe that well (not depicted here)
# so we use an ensemble model, a random forest to predict the open value.
# We use a grid search to look at the best parameters for this model and as
# a scoring we use the log loss metric.
parameters = {'n_estimators': range(100, 301, 50),
              'max_features': ['auto', 'sqrt'],
              'max_depth': [2, 4, 6],
              'min_samples_split': [50, 100, 200]
              }
forest_imp = RandomForestClassifier(random_state=1)
gs_clf = GridSearchCV(forest_imp, parameters,
                      n_jobs=-1, scoring='neg_log_loss', cv=5)
train_data = df[df['open'].notnull()]
X_train = train_data.drop(['id', 'open'], axis=1).values
y_train = train_data['open'].values
gs_clf = gs_clf.fit(X_train, y_train)
print(gs_clf.best_score_)

# %% We take the best model from our grid search and then predict the open
# variable for the test set. We then right the output into a file.
best_forest = gs_clf.best_estimator_
best_forest.fit(X_train, y_train)
test_data = df[df['open'].isnull()]
y_test = best_forest.predict(test_data.drop(['id', 'open'], axis=1).values)
y_test = pd.Series(y_test)
predictions = pd.concat([test_data['id'], y_test], axis=1,
                        names=['id', 'open'])

predictions.to_csv('new_predictions.csv',
                   index=False)
