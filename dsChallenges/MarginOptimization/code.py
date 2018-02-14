#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:55:11 2018

@author: marcduda
"""

import numpy as np
import pandas as pd
import seaborn as sn
import h2o
import matplotlib.pyplot as plt
from h2o.estimators.random_forest import H2ORandomForestEstimator

data = pd.read_csv('aft100k.csv')
#data.drop_duplicates(inplace=True)
print(data.user_operating_system.unique())
print(data.user_device.unique())
print(data[data[['user_device']].isnull().any(axis=1)])
data['isRevenue']=data['revenue']>0
data['roundRevenue']=round(data['revenue'], 2)
data.user_device.fillna("PersonalComputer", inplace=True)
#data["average_seconds_played"] = data.groupby(['user_device','user_operating_system','roundRevenue'])['average_seconds_played'].transform(lambda x: x.fillna(x.mean()))
#data.average_seconds_played.fillna(-1, inplace=True)
print(data.count())
#print(data[data['average_seconds_played']==0].shape)
print(data[data[['average_seconds_played']].isnull().any(axis=1)])

corrMatt = data[["average_seconds_played","cost","revenue"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
def computeMargin(df):
    if df['revenue']==0:
        return 0
    else:
        return (df['revenue']-df['cost'])/df['revenue']
data['margin'] = data.apply(computeMargin,axis=1)
print("The total margin is "+str(data['margin'].sum()))
OS_revenue = data.groupby('user_operating_system').margin.sum()
print(OS_revenue)
profit = data.groupby('user_operating_system').revenue.sum()-data.groupby('user_operating_system').cost.sum()
#%% Modeling
#initiate a h2o cluster and transforme the data into a h2o-specific frame. 
h2o.init(max_mem_size = "2G", nthreads=-1)
col_types=['numeric','categorical','categorical','numeric','numeric','numeric','categorical','numeric','numeric']#
h2ofr = h2o.H2OFrame(data,column_types=col_types)
#split the data into training, testing and validation sets.
splits = h2ofr.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

#set the columns that while be used as features and the ones we want to predict.
y = 'isRevenue'
x = list(h2ofr.columns)
x.remove(y) 
for z in ['creative_id','revenue','roundRevenue','margin']:
    x.remove(z)
#x.remove('average_seconds_played')

#train a random forest regressor on the training set.
RF = H2ORandomForestEstimator(ntrees=200,seed=1,nfolds=15)
RF.train(x=x, y=y, training_frame=train)


#look at the performances of the trained random forest model on the test set.
RF_perf = RF.model_performance(test)
print(RF_perf)

#plot a figure of the variable importances in the trained model. 
fig, ax = plt.subplots()
variables = RF._model_json['output']['variable_importances']['variable']
y_pos = np.arange(len(variables))
scaled_importance = RF._model_json['output']['variable_importances']['scaled_importance']
ax.barh(y_pos, scaled_importance, align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.show()

#print in the console the varable importances.
print(RF._model_json['output']['variable_importances'].as_data_frame())

#h2o.cluster().shutdown()