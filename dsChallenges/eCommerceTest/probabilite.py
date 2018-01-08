#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:41:05 2017

@author: marcduda
"""
#from dateutil import parser
from datetime import datetime
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

#load data and transform the time column
data_purchase = pd.read_csv("purchases.csv")
time = data_purchase['time'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').timestamp())
time = time-time.min()+1
amount = data_purchase['amount']
#get ranges of data
xmin = time.min()
xmax = time.max()
ymin = amount.min()
ymax = amount.max()

#create a grid and calculate the probability on this grid
X_, Y_ = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([X_.ravel(), Y_.ravel()])
values = np.vstack([time, amount])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X_.shape)

#plot the heatmap and the original points
fig, ax = plt.subplots( )
ax.set_xlim([xmin, xmax])
ax.set_xscale("log")
ax.set_ylim([ymin, ymax])
plt.pcolor(X_, Y_, Z, norm=LogNorm(vmin=Z.min(), vmax=Z.max()), cmap='jet')
ax.plot(time, amount, 'k.', markersize=2)
ax.set(adjustable='box-forced')
plt.show()

