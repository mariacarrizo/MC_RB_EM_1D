#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 2_GlobalSearch.py
Description: Script to perform global search inversion for Field case
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 17/12/2023
"""

# Import libraries
import numpy as np
import time
import pandas as pd
import sys
sys.path.insert(1, '../src')

# Load function that performs global search in lookup table
from EM1D import GlobalSearch_2Lay

# Load lookup table and sampling 
LUT = np.load('data/LUTable_2Lay_field.npy')
conds = np.load('data/conds.npy')
thicks =  np.load('data/thicks.npy')

# Load field data 
Dataframe = pd.DataFrame(np.load('data/Field_data.npy'),
                        columns = ['X','Y','Position','Z','H2Q','H4Q','H8Q',
                                   'V2Q','V4Q','V8Q','P2Q','P4Q','P8Q',
                                   'H4IP','H8IP','V4IP','V8IP'])

# Obtain H and V quadrature and in-phase measurements
# For in-phase we only use measurements for offsets > 4 m
data = np.array(pd.concat([Dataframe.loc[:,'H2Q':'V8Q'], Dataframe.loc[:,'H4IP':]], axis=1))

# number of 1D models
npos = len(data)

# Estimate with both Quadrature and In Phase

model_GS = [] # Empty array for estimated model

print('Starting global search ...')

starttime = time.time()
for p in range(npos):
    model_GS.append(GlobalSearch_2Lay(LUT, data[p], conds, thicks, nsl=len(conds)))
endtime = time.time() - starttime

print('Global search Q+IP excution for ', npos, ' positions: ', f"{(endtime/60):.3}", 'minutes')

# Save estimated models
np.save('results/model_2Lay_GS_field', model_GS)
