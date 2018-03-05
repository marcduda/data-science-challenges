# version : Python 3.6.
# This file is divided in three main parts :
# a first one where we load the data and do some preprocessing
# a second where we plot some figures
# and a third where we apply a random forest regressor to predict the count parameter.
# We just need to launch the code to do it all all we can launch the first part and then
# whatever other we want to see (figures or regression)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
import seaborn as sn
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.random_forest import H2ORandomForestEstimator


data = pd.read_csv('data.csv')
# extract the year, month, day of the week and hour from the date variable.
data.datetime = data.datetime.apply(pd.to_datetime)
data['month'] = data.datetime.apply(lambda x: x.month)
data['hour'] = data.datetime.apply(lambda x: x.hour)
data['year'] = data.datetime.apply(lambda x: x.year)
data['day'] = data.datetime.apply(lambda x: x.weekday())
# create a feature that represents the difference between the real temperature and the felt temperature.
data['atempdiff'] = data.temp-data.atemp

#%% Plotting figures
# prepare some lists to use as legends and labels for the plots.
time_factors = ['year', 'season', 'month',
                'day', 'hour', 'holiday', 'workingday']
time_factors_legends = [('2011', '2012'), ('Spring', 'Summer', 'Fall', 'Winter'), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'), ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'),
                        ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'), ('Not Holiday', 'Holiday'), ('Not Working Day', 'Working Day')]
weather_factors = ['weather', 'temp', 'atemp',
                   'atempdiff', 'humidity', 'windspeed', ]
weather_legend = ('Sunny to Cloudy', 'Smoggy',
                  'Light rain \n or snow', 'Heavy rain \n or snow')
categorical = ['weather']

# set parameters for all the figures.
width = 0.35       # the width of the bars
fontP = FontProperties()
fontP.set_size('x-small')  # font of the text

# plot a first figure that shows the time parameters against the registered and casual parameter.
fig, axes = plt.subplots(nrows=2, ncols=4)
for ax, factor, factor_legend in zip(axes.flat[:7], time_factors, time_factors_legends):
    ind = np.arange(0, len(data[factor].unique()))
    rects2 = ax.bar(ind,  data.groupby(
        factor).registered.mean(), width, color='orange')
    rects1 = ax.bar(
        ind+width, data.groupby(factor).casual.mean(), width, color='blue')
    ax.set_ylabel('Number rented bikes')
    ax.set_title(label='Number of rented bikes by ' +
                 factor, fontdict={'fontsize': 9})
    ax.set_yticklabels([])
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels=factor_legend, fontdict={'fontsize': 9})
    if factor in ['month', 'hour']:
        for index, label in enumerate(ax.xaxis.get_ticklabels()):
            if index % 2 != 0:
                label.set_visible(False)
    if factor in ['holiday']:
        ax.legend((rects1[0], rects2[0]), ('Casual rent',
                                           'Registered rent'), loc="upper right", prop=fontP)
    else:
        ax.legend((rects1[0], rects2[0]), ('Casual rent',
                                           'Registered rent'), loc="upper left", prop=fontP)
axes.flat[7].set_visible(False)
plt.suptitle(
    'Influences of time parameters \n on the number of casual and registered rented bikes', fontsize=16)
plt.show()


# plot a second figure that shows the weather parameters against the registered and casual parameter.
fig, axes = plt.subplots(nrows=2, ncols=3)
for ax, factor in zip(axes.flat, weather_factors):
    if factor in categorical:
        ind = np.arange(0, len(data[factor].unique()))
        rects2 = ax.bar(ind,  data.groupby(
            factor).registered.mean(), width, color='orange')
        rects1 = ax.bar(
            ind+width, data.groupby(factor).casual.mean(), width, color='blue')
        # add some text for labels, title and axes ticks.
        ax.set_ylabel('Number rented bikes')
        ax.set_title(label='Number of rented bikes by ' +
                     factor, fontdict={'fontsize': 9})
        ax.set_yticklabels([])
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(labels=weather_legend, fontdict={'fontsize': 9})

        ax.legend((rects1[0], rects2[0]), ('Casual rent',
                                           'Registered rent'), loc="upper right", prop=fontP)
    else:
        ind = np.sort(data[factor].unique())
        rects2 = ax.bar(ind,  data.groupby(
            factor).registered.mean(), color='orange')
        rects1 = ax.bar(ind,  data.groupby(factor).casual.mean(), color='blue')
        ax.set_ylabel('Average number rented bikes')
        ax.set_title(label='Average number of rented bikes by ' +
                     factor, fontdict={'fontsize': 9})
        ax.set_yticklabels([])
        if factor in ['temp', 'atempdiff', 'atemp']:
            ax.legend((rects1, rects2), ('Casual rent',
                                         'Registered rent'), loc="upper left", prop=fontP)
        else:
            ax.legend((rects1, rects2), ('Casual rent',
                                         'Registered rent'), loc="upper right", prop=fontP)
plt.suptitle(
    'Influences of weather parameters \n on the number of casual and registered rented bikes', fontsize=16)
plt.show()


# plot a third figure that shows the time parameters against the count parameter.
fig, axes = plt.subplots(nrows=2, ncols=4)
for ax, factor, factor_legend in zip(axes.flat[:7], time_factors, time_factors_legends):
    ind = np.arange(0, len(data[factor].unique()))
    rects2 = ax.bar(ind,  data.groupby(factor)[
                    'count'].mean(), width, color='green')
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Average number rented bikes')
    ax.set_title(label='Average number of rented bikes by ' +
                 factor, fontdict={'fontsize': 9})
    ax.set_yticklabels([])
    ax.set_xticks(ind)
    ax.set_xticklabels(labels=factor_legend, fontdict={'fontsize': 9})
    if factor in ['month', 'hour']:
        for index, label in enumerate(ax.xaxis.get_ticklabels()):
            if index % 2 != 0:
                label.set_visible(False)
    if factor in ['holiday']:
        ax.legend([rects2[0]], ['Total rent'], loc="upper right", prop=fontP)
    else:
        ax.legend([rects2[0]], ['Total rent'], loc="upper left", prop=fontP)
axes.flat[7].set_visible(False)
plt.suptitle(
    'Influences of time parameters \n on the total number of rented bikes', fontsize=16)
plt.show()

# plot a fourth figure that shows the weather parameters against the count parameter.
fig, axes = plt.subplots(nrows=2, ncols=3)
for ax, factor in zip(axes.flat, weather_factors):
    if factor in categorical:
        ind = np.arange(0, len(data[factor].unique()))
        rects2 = ax.bar(ind,  data.groupby(factor)[
                        'count'].mean(), width, color='green')
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Average number rented bikes')
        ax.set_title(label='Average number of rented bikes by ' +
                     factor, fontdict={'fontsize': 9})
        ax.set_yticklabels([])
        ax.set_xticks(ind)
        ax.set_xticklabels(labels=weather_legend, fontdict={'fontsize': 9})

        ax.legend([rects2[0]], ['total rent'], loc="upper right", prop=fontP)
    else:
        ind = np.sort(data[factor].unique())
        # ax.scatter(data[factor],data['count'],color="green")#
        rects2 = ax.bar(ind,  data.groupby(factor)[
                        'count'].mean(), color='green')
        ax.set_ylabel('Average number rented bikes')
        ax.set_title(label='Average number of rented bikes by ' +
                     factor, fontdict={'fontsize': 9})
        ax.set_yticklabels([])
        if factor in ['temp']:
            ax.legend([rects2], ['Total rent'], loc="upper left", prop=fontP)
        else:
            ax.legend([rects2], ['Total rent'], loc="upper right", prop=fontP)
plt.suptitle(
    'Influences of weather parameters \n on the total number of rented bikes', fontsize=16)


# plot a fifth figure that shows the correlations among the quantitative parameters.
corrMatt = data[["temp", "atemp", "casual", "registered",
                 "humidity", "windspeed", "count", "atempdiff"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
sn.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)

#%% Modeling
# initiate a h2o cluster and transforme the data into a h2o-specific frame.
h2o.init(max_mem_size="2G", nthreads=-1)
col_types = ['time', 'categorical', 'categorical', 'categorical', 'categorical',
             'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric',
             'categorical', 'categorical', 'categorical', 'categorical', 'numeric']
h2ofr = h2o.H2OFrame(data, column_types=col_types)
# split the data into training, testing and validation sets.
splits = h2ofr.split_frame(ratios=[0.7, 0.15], seed=1)
train = splits[0]
valid = splits[1]
test = splits[2]

# set the columns that while be used as features and the ones we want to predict.
y = 'count'
x = list(h2ofr.columns)
x.remove(y)  # remove response variable
for z in ['datetime', 'temp', 'casual', 'registered']:
    x.remove(z)

# train a random forest regressor on the training set.
RF = H2ORandomForestEstimator(ntrees=200, seed=1, nfolds=5)
RF.train(x=x, y=y, training_frame=train)
#grid_search = H2OGridSearch(H2ORandomForestEstimator, hyper_params=hyper_parameters)
#grid_search.train(x, y, training_frame=train)
# grid_search.show()
#RF_perf = RF.model_performance(train)
# print(RF_perf)

# look at the performances of the trained random forest model on the test set.
RF_perf = RF.model_performance(test)
print(RF_perf)

# plot a (sixth) figure of the variable importances in the trained model.
fig, ax = plt.subplots()
variables = RF._model_json['output']['variable_importances']['variable']
y_pos = np.arange(len(variables))
scaled_importance = RF._model_json['output']['variable_importances']['scaled_importance']
ax.barh(y_pos, scaled_importance, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.show()

# print in the console the varable importances.
print(RF._model_json['output']['variable_importances'].as_data_frame())

# when the model seems to be tuned the best we can, look at the performances on
# a validation set that was never presented to the model to see if we didn't overfit our
# model on the test set.
RF_perf = RF.model_performance(test)
print(RF_perf)


# shut down the h2o cluster
h2o.cluster().shutdown()
