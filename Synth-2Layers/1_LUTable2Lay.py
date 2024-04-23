# Script for generating 2-layered lookup table

# import libraries
import numpy as np
from empymod import filters
import time
from joblib import Parallel, delayed
import sys
sys.path.insert(1, '../src')

# Import forward function for 2-layered models
from EM1D import EMf_2Lay_HVP

# Define hankel filter
filt = filters.key_201_2012() 

# Define EMI instrument geometry
offsets = np.array([2, 4, 8]) # in meters
height = 0.10 # meters height From ground surface to center of coils
freq = 9000 # Frequency in Hz

# Lambda numbers to calculate forward
lambd = filt.base/offsets[:,np.newaxis] 

# Store survey details
survey = {'offsets': offsets,
          'height': height,
          'freq': freq,
          'lambd': lambd,
          'filt': filt}

np.save('data/survey_2Lay.npy', survey)

# sampling of conductivities
nsl = 101 # number of samples
s0 = -2 # minimum conductivity in S/m
s1 = 0.3 # maximum conductivity in S/m
# conductivities array
conds = np.logspace(s0, s1, nsl)

# Sampling of 1st layer thickness
th0 = 0.1 # minimum thickness in m
th1 = 7   # maximum thickness in m
# thickness array
h_1 = np.linspace(th0, th1, nsl)

# Generate lookup table
print('Start generating Lookup table for 2-layered models')
startTime = time.time()

DataEM = Parallel(n_jobs=-1,verbose=1)(delayed(EMf_2Lay_HVP)(lambd, sigma1, sigma2, h1, 
                  height, offsets, freq, filt) for sigma1 in conds for sigma2 in conds for h1 in h_1)

executionTime = ((time.time() - startTime))/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

# Save the table, sampling and models

np.save('data/LUTable_2Lay', DataEM)
np.save('data/conds', conds)
np.save('data/thicks', h_1)

