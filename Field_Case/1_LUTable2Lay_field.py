#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: 1_LUTable2Lay_field.py
Description: Script for generating 2-layered lookup table for Field case
Author: @mariacarrizo
Email: mecarrizomasca@tudelft.nl
Date created: 15/12/2023
"""

# import libraries
import numpy as np
from empymod import filters
import time
from joblib import Parallel, delayed
import sys
sys.path.insert(1, '../src')

# Import forward function for 2-layered models in Field case
from EM1D import EMf_2Lay_HV_field

# Define hankel filter
filt = filters.key_201_2012() 

# Define EMI instrument geometry
offsets = np.array([2, 4, 8]) # in meters
height = 0.47 # meters height from ground surface to center of coils
freq = 9000 # Frequency in Hz

# Lambda numbers
lambd = filt.base/offsets[:,np.newaxis] 

# Store survey details
survey = {'offsets': offsets,
          'height': height,
          'freq': freq,
          'lambd': lambd,
          'filt': filt}

np.save('data/survey_field.npy', survey)

# sampling of conductivities
nsl = 71 # number of samples
s0 = -2 # minimum conductivity in S/m
s1 = -0.6 # maximum conductivity in S/m
# conductivities array
conds = np.logspace(s0, s1, nsl)

# Sampling of 1st layer thickness
th0 = 0.1 # minimum thickness in m
th1 = 6   # maximum thickness in m
# thickness array
thicks = np.linspace(th0, th1, nsl)

# Generate lookup table

startTime = time.time()

LUT = Parallel(n_jobs=-1,verbose=1)(delayed(EMf_2Lay_HV_field)(lambd, sigma1, sigma2, h1, 
                  height, offsets, freq, filt) for sigma1 in conds for sigma2 in conds for h1 in thicks)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

# Save the table, sampling and models
np.save('data/LUTable_2Lay_field', LUT)
np.save('data/conds', conds)
np.save('data/thicks', thicks)

