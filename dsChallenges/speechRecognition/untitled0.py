#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:29:46 2017

@author: marcduda
"""

import scipy.io.wavfile
from scipy import signal
import soundfile as sf
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import pylab
from pywt import WaveletPacket
import pywt


def merge_intervals(intervals):
    s = sorted(intervals, key=lambda t: t[0])
    m = 0
    for t in s:
        if t[0] > s[m][1]:
            m += 1
            s[m] = t
        else:
            s[m] = (s[m][0], t[1])
    return s[:m+1]


files = os.listdir("vad_data/")
json_files = [file for file in files if "json" in file]
wav_files = [file for file in files if "wav" in file]

good_size = len(wav_files)+len(json_files) == len(files)
global_labels = []
global_dataset = None
level1 = None
for file in wav_files[:1]:  # [:100]
    print("processing file number " +
          str(wav_files.index(file)+1)+" on "+str(len(wav_files)))
    file_num = file.split(".")[0]
    rate, data = scipy.io.wavfile.read("vad_data/"+file)
    noise_power = 0.1 * rate / 2
    time = np.arange(len(data)) / float(rate)
    noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    data = data+noise
    #data1, samplerate = sf.read("vad_data/"+file)

#    plt.figure()
#    wp = WaveletPacket(data, 'db4', maxlevel=4)
#    pylab.bone()
#    pylab.subplot(wp.maxlevel + 1, 1, 1)
#    pylab.plot(data, 'k')
#    pylab.xlim(0, len(data) - 1)
#    pylab.title("Wavelet packet coefficients")
#    level1 = wp.get_level(1, "freq")
#    level1.reverse()
#    #level1=-abs(np.array([n.data for n in level1]))
#
#    for i in range(1, wp.maxlevel + 1):
#        ax = pylab.subplot(wp.maxlevel + 1, 1, i + 1)
#        nodes = wp.get_level(i, "freq")
#        nodes.reverse()
#        labels = [n.path for n in nodes]
#        values = -abs(np.array([n.data for n in nodes]))
#        pylab.imshow(values, interpolation='nearest', aspect='auto')
#        pylab.yticks(np.arange(len(labels) - 0.5, -0.5, -1), labels)
#        pylab.setp(ax.get_xticklabels(), visible=False)
#    pylab.show()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(data)
    plt.title('Audio signal with noise, in red the parts identified as voice')

    #plt.subplot(2, 1, 2)
    # plt.plot(data1)
    # plt.show()
    #file_length = len(data)/16000
    json_file = json.loads(
        open("vad_data/"+file_num+".json").read())['speech_segments']
    speech_intervals = [(dict['start_time'], dict['end_time'])
                        for dict in json_file]
    speech_intervals = merge_intervals(speech_intervals)
    speech_intervals = [(round(intervals[0]*rate), round(intervals[1]*rate))
                        for intervals in speech_intervals]
    for intervals in speech_intervals:
        plt.plot(range(intervals[0], intervals[1]),
                 data[intervals[0]:intervals[1]], 'r')
        # plt.axvline(x=intervals[0],color='r')
        # plt.axvline(x=intervals[1],color='r')
    #plt.axis([0, len(data),min(data),max(data)])

    # for i in range(0,len(data),320):
    #    plt.axvline(x=i,color='black')

    plt.margins(0)
    plt.show()
    plt.subplot(3, 1, 2)  # plt.figure()
    f, t, Zxx = signal.stft(data, rate)  # , nperseg=1000
    coef, freqs = pywt.cwt(data, np.arange(
        1, 10), 'mexh', sampling_period=1/rate)
    # plt.matshow(coef)
    plt.pcolormesh(time, freqs, coef)
    plt.title('Coefficients of the CWT')
    # plt.pcolormesh(t, f, np.abs(Zxx))#, vmin=0, vmax=amp

    # for i in t[::3]:
    #    plt.axvline(x=i,color='black')
    plt.show()
    #intervals_time = list(zip(t[::3],t[::3][1:]))
    #intervals_time = [(a*rate,b*rate) for (a,b) in list(zip(t[::3],t[::3][1:]))]
    window_cwt = int(0.02*rate)
    intervals_time_cwt = list(zip(time[::window_cwt], time[::window_cwt][1:]))
    intervals_time_cwt = [(a*rate, b*rate) for (a, b) in intervals_time_cwt]
    labels = []
    data_stft = None
    j = 0
#    #Zxx = Zxx.T
    power = np.abs(Zxx * np.conj(Zxx))
    plt.subplot(3, 1, 3)
    plt.pcolormesh(t, f, power)
    plt.title('Power of the STFT')
    plt.show()
#    for (start,end) in intervals_time_cwt:#intervals_time
#        speech_label_found = False
#        label = 0
#        i=0
#        (start,end) = (int(start),int(end))
#        while (not speech_label_found) & (i<len(speech_intervals)):
#            (start_sp,end_sp)= speech_intervals[i]
#            if end_sp-start_sp<end-start:
#                speech_label_found = True
#                label  = int((min(end,end_sp)-max(start,start_sp))/(end-start)>0.5)
#            else:
#                if end<=start_sp:
#                    speech_label_found = True
#                elif ((end>start_sp) & (start<start_sp))|((end>end_sp) & (start<=end_sp)):
#                    speech_label_found = True
#                    label  = int((min(end,end_sp)-max(start,start_sp))/(end-start)>0.5)
#                elif (end<=end_sp) & (start>=start_sp):
#                    speech_label_found = True
#                    label = 1
#            i+=1
#        global_labels.append(label)#labels
#        if data_stft is None:
#            data_stft = coef[:,j:j+window_cwt].reshape(1,len(freqs),window_cwt,1)#power[:,j:j+3].reshape(1,len(f)*3)
#        else:
#            current_data = coef[:,j:j+window_cwt].reshape(1,len(freqs),window_cwt,1)#power[:,j:j+3].reshape(1,len(f)*3)
#            data_stft = np.append(data_stft,current_data,axis=0)
#        j+=window_cwt#3
#    if global_dataset is None:
#        global_dataset = data_stft
#    else:
#        global_dataset = np.append(global_dataset,data_stft,axis=0)
    # global_labels.append(labels)

    # data_stft


#    plt.subplot(3, 1, 3)
#    f, t, Sxx = signal.spectrogram(data, rate)
#    plt.pcolormesh(t, f, Sxx)

#%%
#name_dataset = "datasetCWTWithNoiseFirst1002D"
#name_label = "labelsCWTWithNoiseFirst1002D"
#np.save(name_dataset, global_dataset)
#np.save(name_label, global_labels)
