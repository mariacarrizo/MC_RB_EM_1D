### Code that creates Lookup table for a 3-layered 1D models

## Import libraries
import numpy as np
import time
from joblib import Parallel, delayed
from empymod import filters
import sys
path = '../src'
sys.path.insert(0, path)

# Import forward function for 3-layered 1D models
from EM1D import EMf_3Lay_HVP_field

# set here the number of workers to compute
# if set to -1 will take all the workers available
n_workers = 8

# Define hankel filter
filt = filters.key_201_2012() 

# Define EMI instrument geometry
offsets = np.array([2, 4, 8]) # in meters
height = 0.47 # meters height From ground surface to center of coils
freq = 9000 # Frequency in Hertz

# Lambda numbers
lambd = filt.base/offsets[:,np.newaxis] 

# Store survey details
survey = {'offsets': offsets,
          'height': height,
          'freq': freq,
          'lambd': lambd,
          'filt': filt}

np.save('data/survey_3Lay.npy', survey)

# sampling of conductivities
nsl = 51 # number of samples
s0 = -2 # minimum conductivity in S/m
s1 = -0.6 # maximum conductivity in S/m
# conductivities array
conds = np.logspace(s0, s1, nsl)

# Sampling of 1st layer thickness
th0 = 0.1 # minimum thickness in m
th1 = 7   # maximum thickness in m
# thickness array
thicks = np.linspace(th0, th1, nsl)

# Generate lookup table
print('Start calculating Lookup table ...')
startTime = time.time()

LUT = Parallel(n_jobs=n_workers,verbose=0)(delayed(EMf_3Lay_HVP_field)(lambd, sigma1, sigma2, sigma3, h1, h2, 
                  height, offsets, freq, filt) for sigma1 in conds for sigma2 in conds 
                  for sigma3 in conds for h1 in thicks for h2 in thicks)

executionTime = ((time.time() - startTime))/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

# Save the table, sampling and models
np.save('data/LUTable_3Lay_field', LUT)
np.save('data/conds', conds)
np.save('data/thicks', thicks)



