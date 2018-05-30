# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:41:37 2018

@author: marc
"""
# The aim is to build a model that can predict the willingness of a client to pay to have its package shipped.
# First we need to determine what the willingness is. We interpreted it as the difference from of price between 
# what the client payed and the mean price payed for a similar package. 

import pandas as pd


def getKey(item):
    return item[1]


data = pd.read_csv('dataset.csv')
print(data.describe())
print(data.count())
data.drop(['FlownYear'], axis=1, inplace=True)
data.drop(['DestinationCode'], axis=1, inplace=True)

# %% data cleaning
data = data[data['Revenue'] != 0]
codeName = data['AgentCode']+" "+data['AgentName']
codeName = codeName.unique()
code = [agent.split()[1] for agent in codeName]
doubleCode = [[x, code.count(x)] for x in code]
output = []
for x in doubleCode:
    if x not in output:
        output.append(x)
doubles = sorted(output, key=getKey, reverse=True)
doublesName = [double[0] for double in doubles if double[1] > 1]
print(doublesName)
codeAndName = [l.split() for l in codeName]
codeAndName = [l for l in codeAndName if l[1] in doublesName]
outputDict = {}
for x in codeAndName:
    if x[1] not in outputDict.keys():
        outputDict[x[1]] = x[0]
print(outputDict)
print(len(data['AgentCode'].unique()))
print(len(data['AgentName'].unique()))


def changeAgentCode(df):
    if df['AgentName'] in outputDict.keys():
        df['AgentCode'] = outputDict[df['AgentName']]
    else:
        df['AgentCode'] = df['AgentCode']
    return df


data = data.apply(changeAgentCode, axis=1)
print(len(data['AgentCode'].unique()))
print(len(data['AgentName'].unique()))


def revenuePerPiece(df):
    if df['Pieces'] == 0:
        return 0
    else:
        return df['Revenue']/df['Pieces']


def weightPerPiece(df):
    if df['Pieces'] == 0:
        return 0
    else:
        return df['ChargeableWeight']/df['Pieces']


# Continuous values binning
data['WeightPerPiece'] = data.apply(weightPerPiece, axis=1)
data['BinnedWeight'] = data['WeightPerPiece'].apply(lambda x: 0 if x < 100 else (1 if x < 1000 else 2))
data['BinnedPieces'] = data['Pieces'].apply(lambda x: 0 if x < 5 else (1 if x < 20 else 2))
data['RevenuePerPiece'] = data.apply(revenuePerPiece, axis=1)

# data grouping by shipment type
dfgroup = data.groupby(['FlownMonth', 'OriginCode',
                        'CargoType', 'ProductCode', 'SpecialHandlingCodeList',
                        'CommodityCode', 'BinnedWeight', 'BinnedPieces'])
revMeans = dfgroup.RevenuePerPiece.mean()


def revenueToMean(df, means):
    return df['RevenuePerPiece']-means.loc[df['FlownMonth'], df['OriginCode'],
                                           df['CargoType'],
                                           df['ProductCode'], df['SpecialHandlingCodeList'],
                                           df['CommodityCode'], df['BinnedWeight'],
                                           df['BinnedPieces']]


# creation of a variable indication the difference of revenue between the
# shipment and the mean revenue of its group
data['Willingness'] = data.apply(lambda x: revenueToMean(x, revMeans), axis=1)

# data grouping by shipment type and agent code
agentgroup = data.groupby(['AgentCode', 'FlownMonth', 'OriginCode',
                           'CargoType', 'ProductCode',
                           'SpecialHandlingCodeList', 'CommodityCode',
                           'BinnedWeight', 'BinnedPieces'])

# series that contains the mean difference of revenue for an agent and for a
# shipment type
agentWillingness = agentgroup.Willingness.mean()
# %%
#col = ['ChargeableWeight', 'Pieces', 'Revenue', 'WeightPerPiece', 'RevenuePerPiece']
col = ['ChargeableWeight', 'BinnedWeight', 'BinnedPieces', 'Revenue', 'RevenuePerPiece']
data1 = data.copy(deep=True)
for i in col:
    data1.drop([i], axis=1, inplace=True)
data.to_csv('dataProcessed.csv')

# %%
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
trace = go.Scatter(
    x = data['WeightPerPiece'].values,
    y = data['RevenuePerPiece'].values,
    mode = 'markers')
d = [trace]
# Plot and embed in ipython notebook!
#py.iplot(d, filename='basic-scatter')
#plot_url = py.plot(d, filename='basic-line')
fig, ax = plt.subplots()
plt.plot(data['BinnedPieces'].values, data['RevenuePerPiece'], 'b.')
plt.show()
fig, ax = plt.subplots()
plt.plot(data['BinnedWeight'].values, data['RevenuePerPiece'], 'b.')
plt.show()

# %%
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
categories = ['DocumentRatingSource', 'FlownMonth', 'AgentCode', 'AgentName',
              'POS', 'POSCountryName', 'OriginCode', 'CargoType',
              'ProductCode', 'SpecialHandlingCodeList', 'CommodityCode']#,
              #'BinnedWeight', 'BinnedPieces']
for category in categories:
    data1[category] = data1[category].astype('category')
    data1[category] = data1[category].cat.codes
h2o.init(max_mem_size="5G", nthreads=-1)
data1 = data1[['WeightPerPiece','Pieces','RevenueToMean']]
col_types = ['numeric', 'numeric', 'numeric']
#col_types = ['categorical', 'categorical', 'categorical', 'categorical',
#             'categorical', 'categorical', 'categorical', 'categorical',
#             'categorical', 'categorical', 'categorical', 'numeric',
#             'numeric', 'numeric']
h2ofr = h2o.H2OFrame(data1, column_types=col_types)

# Split the data into training, testing and validation sets.
splits = h2ofr.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

# Set the columns that will be used as features and the one we want to predict.
y = 'Willingness'
x = list(h2ofr.columns)
x.remove(y)

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
                nfolds=7, #distribution='AUTO',
                seed=1, fold_assignment='Modulo',
                keep_cross_validation_predictions=True
                )
# Get the grid results, sorted by validation RMSLE.
gbm_gridperf1 = gbm_grid1.get_grid(sort_by='rmse', decreasing=False)
print(gbm_gridperf1)
