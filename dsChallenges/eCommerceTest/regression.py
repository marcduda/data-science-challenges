#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:25:04 2017

@author: marcduda
"""

import numpy as np
from scipy.signal import wavelets
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#load data
data = np.genfromtxt ('regression_data.csv', delimiter=",",skip_header=1)
x= data[:,0]
y = data[:,1]

#compute fft of the signal
DATA = np.fft.fft(y)
print(np.argmax(DATA))
#%%


#generate a morlet wavelet 
morlet= wavelets.morlet(len(x), 7,0.6, complete=True).real
morlet = np.append(morlet[90:],morlet[1000]*np.ones(90))

#plot the points and the fitting function
plt.scatter(x,y)
plt.show
plt.scatter(x,1.6*morlet)
plt.show

print(r2_score(1.6*morlet, y))
print(np.sqrt(sum(pow(1.6*morlet-y,2))))
#%%
DATA1 = np.fft.fft(1.6*morlet)
print(np.argmax(DATA1.real))
plt.plot(DATA1.real)
res = DATA1.real

