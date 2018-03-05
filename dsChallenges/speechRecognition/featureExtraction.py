#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:30:40 2017

@author: marcduda
"""

import scipy.io.wavfile
from scipy import signal
import os
import json
import numpy as np


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
for file in wav_files:
    print("processing file number " +
          str(wav_files.index(file)+1)+" on "+str(len(wav_files)))
    file_num = file.split(".")[0]
    rate, data = scipy.io.wavfile.read("vad_data/"+file)
    noise_power = 0.5 * rate / 2
    time_raw = np.arange(len(data)) / float(rate)
    noise = np.random.normal(scale=np.sqrt(noise_power), size=time_raw.shape)
    data = data+noise

    json_file = json.loads(
        open("vad_data/"+file_num+".json").read())['speech_segments']
    speech_intervals = [(dict['start_time'], dict['end_time'])
                        for dict in json_file]
    speech_intervals = merge_intervals(speech_intervals)
    speech_intervals = [(round(intervals[0]*rate), round(intervals[1]*rate))
                        for intervals in speech_intervals]
    f, t, Zxx = signal.stft(data, rate)
    window_cwt = int(0.02*rate)
    window_stft = 3
    window_raw = int(0.02*rate)
    window = window_stft
    time = t
    intervals_time = list(zip(time[::window], time[::window][1:]))
    intervals_time = [(a*rate, b*rate) for (a, b) in intervals_time]
    data_file = None
    j = 0
    power = np.abs(Zxx * np.conj(Zxx))
    for (start, end) in intervals_time:
        speech_label_found = False
        label = 0
        i = 0
        (start, end) = (int(start), int(end))
        while (not speech_label_found) & (i < len(speech_intervals)):
            (start_sp, end_sp) = speech_intervals[i]
            if end_sp-start_sp < end-start:
                speech_label_found = True
                label = int(
                    (min(end, end_sp)-max(start, start_sp))/(end-start) > 0.5)
            else:
                if end <= start_sp:
                    speech_label_found = True
                elif ((end > start_sp) & (start < start_sp)) | ((end > end_sp) & (start <= end_sp)):
                    speech_label_found = True
                    label = int(
                        (min(end, end_sp)-max(start, start_sp))/(end-start) > 0.5)
                elif (end <= end_sp) & (start >= start_sp):
                    speech_label_found = True
                    label = 1
            i += 1

        if data_file is None:
            data_file = power[:, j:j+window].reshape(1, len(f), window)
            global_labels.append(label)
        else:
            current_data = power[:, j:j+window].reshape(1, len(f), window)
            if current_data.shape == (1, len(f), window):
                data_file = np.append(data_file, current_data, axis=0)
                global_labels.append(label)
            else:
                print(str(j))
        j += window
    if global_dataset is None:
        global_dataset = data_file
    else:
        global_dataset = np.append(global_dataset, data_file, axis=0)

#%%
name_dataset = "datasetSTFTWithNoise2D"
name_label = "labelsSTFTWithNoise2D"
np.save(name_dataset, global_dataset)
np.save(name_label, global_labels)
