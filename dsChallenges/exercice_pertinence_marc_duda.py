# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:41:16 2017

@author: marc
"""
# The main idea is to read each line one by one and increment the "pertinence"
# as long as the integer on the current line is greater or equal to zero.
# If not we look at the current line and the 4 next lines to see
# if there is a potential candidate that won't diminish the "pertinence"
# ie a line that has a value greater or equal to zero (we stop at the first we find),
# if we find one we go straight to it and jump over the lines in between.
# If there isn't a suitable candidate, we choose to go to the line that has the least negative value
# Side effect: if the least negative value is on the current line, the algorithm stays in place
# so to prevent that we look only at the 4 next lines the next time

import numpy as np
import re

with open("offres.txt") as f:
    content = f.readlines()
    content = [x.strip() for x in content]
    content = [x.strip('\n') for x in content]
    content = [x for x in content if not re.match(r'^\s*$', x)]
    pertinence = 0
    index = 0
    stayStill = False
    while index < len(content):  # read the lines of the input
        # the current value is greater or equal to zero all ok we go to the next line
        if int(content[index]) >= 0:
            pertinence += int(content[index])
            if len(content)-index < 4 and sum(1 for value in content[index:] if int(value) >= 0) <= 0:
                index = len(content)
            else:
                index += 1

        else:  # the current value is negative, we look for a more suitable line
            if not(stayStill):
                foundCandidate = False
                next = 0
                # search suitable line among the four next
                while (not(foundCandidate) and next <= 5 and (index+next) < len(content)):
                    foundCandidate = int(content[index+next]) >= 0
                    next += 1
                next = next-1
                # a suitable line is not found, we take the "least bad" among the 5 consecutive lines
                if not(foundCandidate):
                    maxSlice = min(index+5, len(content))
                    sliceContent = [int(x) for x in content[index:maxSlice]]
                    if maxSlice == len(content):
                        index = len(content)
                    else:
                        pertinence += max(sliceContent)
                        index += np.argmax(sliceContent)
                        if np.argmax(sliceContent) == 0:
                            stayStill = True
                else:  # a more suitable line is found, we go directly there
                    index += next
            else:  # case if we were already on this line to search for suitable lines
                maxSlice = min(index+5, len(content))
                sliceContent = [int(x) for x in content[index+1:maxSlice]]
                pertinence += max(sliceContent)
                index += np.argmax(sliceContent)
                stayStill = False
