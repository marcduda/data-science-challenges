#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:49:47 2018

@author: marcduda
"""
import pandas as pd
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

# The name of the columns has been changed for anonymity purposes.
# The data appear to be quite clean and have a lot of categorical data.
# The effect of each feature on the churn was priorly analysed with Tableau.
# The aim is to predict if a client will unsubscribe from a phone contract
# based on information on the client.

# Import the training and testing sets and concatenate it to one dataset
# to preprocess it the same way.
train_data = pd.read_csv('train.csv', sep=";")
train_data['test'] = 0
test_data = pd.read_csv('test.csv', sep=";")
test_data['test'] = 1
data = pd.concat([train_data, test_data])

# Fill in the missing the missing values in the feature16 columns, in this
# case the ones for which the client subscribed less than a month ago.


def is_float(input):
    try:
        num = float(input)
    except ValueError:
        return False
    return True


def computeDiffTotalCharges(data):
    x = data['feature16']
    if not is_float(x):
        x = float(data['feature17'])*float(data['feature18'])
    return float(x)


data['feature16'] = data.apply(computeDiffTotalCharges, axis=1)

# Transform the categorical features into actual categorical columns and
# encode the values with numbers.
categories = ['feature1', 'feature2', 'feature3', 'feature4',
              'feature5', 'feature6', 'feature7',
              'feature8', 'feature9', 'feature10',
              'feature11', 'feature12', 'feature13',
              'feature14', 'feature15']
for category in categories:
    data[category] = data[category].astype('category')
    data[category] = data[category].cat.codes

# Resplit the data into trainin and testing set.
train_dt = data[data['test'] == 0]
test_dt = data[data['test'] == 1]

# %% Modeling

# Initiate a h2o cluster and transforme the data into a h2o-specific frame.
h2o.init(max_mem_size="5G", nthreads=-1)
col_types = ['numeric', 'categorical', 'categorical', 'categorical',
             'categorical', 'numeric', 'categorical', 'categorical',
             'categorical', 'categorical', 'categorical', 'categorical',
             'categorical', 'categorical', 'categorical', 'categorical',
             'categorical', 'categorical', 'numeric', 'numeric',
             'categorical', 'categorical']
h2ofr = h2o.H2OFrame(train_dt, column_types=col_types)

# Split the data into training, testing and validation sets.
splits = h2ofr.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

# Set the columns that will be used as features and the one we want to predict.
y = 'Churn'
x = list(h2ofr.columns)
x.remove(y)
x.remove('test')
x.remove('id')

# Set the hyperparameters to look at in the grid search.
hyper_params = {'learn_rate': [0.1, 0.15, 0.2],
                'max_depth': [9, 17, 20],
                'ntrees': [200, 500, 800],
                'col_sample_rate': [0.4, 0.6, 0.8]}

# Search criteria
search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 5, 'seed': 1}

# Set a grid search with with a GBM model and train it on the h2o training set.
gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          # grid_id='gbm_grid1',
                          hyper_params=hyper_params,
                          search_criteria=search_criteria
                          )
gbm_grid1.train(x=x, y=y,
                training_frame=train,
                validation_frame=valid,
                nfolds=7, distribution='bernoulli',
                seed=1, fold_assignment='Modulo',
                keep_cross_validation_predictions=True
                )
# Get the grid results, sorted by validation logloss.
gbm_gridperf1 = gbm_grid1.get_grid(sort_by='logloss', decreasing=False)
print(gbm_gridperf1)

# Grab the top GBM model, chosen by validation logloss.
best_gbm1 = gbm_gridperf1.models[0]

# Evaluate the model performance on a test set
# to get an honest estimate of the top model performance.
best_gbm_perf1 = best_gbm1.model_performance(test)

# Since the model generalizes well, we can retrain it on the whole data.
model_GBM = best_gbm1
model_GBM.train(x=x, y=y,
                training_frame=h2ofr)

# %% Predict the Churn on the test set and write it into a file.
test_fr = h2o.H2OFrame(test_dt, column_types=col_types)
for i in ['id', 'Churn', 'test']:
    test_fr = test_fr.drop(i)
GBM_pred = h2o.as_list(model_GBM.predict(test_fr))
GBM_pred = pd.concat([test_dt[['id']], GBM_pred[['1']]], axis=1)
GBM_pred.to_csv("predict.csv", index=False, header=['id', 'output'], sep=";")

# Shut down the h2o cluster.
h2o.cluster().shutdown()
