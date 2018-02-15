#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:56:17 2018

@author: marcduda
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.patches as patches
import re
import enchant

d = enchant.Dict("fr_FR")

directory = "statuts-ocr/"
files = os.listdir(directory)

def get_spaced_colors(n):
    max_value = 16581375 
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    
    return [(int(i[:2], 16)/255, int(i[2:4], 16)/255, int(i[4:], 16)/255,1) for i in colors]

list_all_paragraphe = []
file = files[:1]
if os.path.getsize(directory+file) > 0:
    ocr_txt = pd.read_csv(directory+file,header=None, delim_whitespace=True,
                          quoting=csv.QUOTE_NONE, encoding='utf-8')
    ocr_txt.columns = ['page','x0','y0','x1','y1','word']
    
    ocr_txt['paragraphe'] = 0
    paragraphe = 0
    current_string = ""
    for i in range(1, len(ocr_txt)):
        word = re.sub(r'[^\w\s]',' ',str(ocr_txt.iloc[i, ocr_txt.columns.get_loc('word')]) )
        if abs(ocr_txt.iloc[i, ocr_txt.columns.get_loc('y0')]-ocr_txt.iloc[i-1, ocr_txt.columns.get_loc('y0')])>30: 
            paragraphe+=1
            list_all_paragraphe.append(current_string)
            if d.check(word):
                current_string =word
        elif d.check(word):
            current_string+=" "+word
        ocr_txt.iloc[i, ocr_txt.columns.get_loc('paragraphe')]=paragraphe

    colors = get_spaced_colors(60)
    ocr_head = ocr_txt[:750].values
    for j in np.unique(ocr_head[:,0]):
        ocr_head_loc = ocr_head[ocr_head[:,0]==j]
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        plt.scatter(ocr_head_loc[:,1], -ocr_head_loc[:,2],s=1)
        for p in [patches.Rectangle(
                    (ocr_head[i,1], -ocr_head[i,2]),
                    (ocr_head[i,3]-ocr_head[i,1]),
                    -(ocr_head[i,4]-ocr_head[i,2]),
                    color=colors[ocr_head[i,6]]
                ) for i in range(0,len(ocr_head)) if ocr_head[i,0]==j
            ]:
                ax.add_patch(p)


    