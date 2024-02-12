# Script to perform global search inversion for Field case

# Import libraries
import numpy as np
import time
import sys
sys.path.insert(1, '../../src')

# Load function that performs global search in lookup table
from EM1D import GlobalSearch_3Lay

# Load lookup table and sampling 
LUT = np.load('data/LUTable_3Lay.npy')
conds = np.load('data/conds.npy')
thicks =  np.load('data/thicks.npy')

# Load field data 
# Obtain H and V quadrature and in-phase measurements
# For in-phase we only use measurements for offsets > 4 m
data = np.load('data/Field_data.npy')

# number of 1D models
npos = len(data)

# Estimate with both Quadrature and In Phase
model_est = [] # Empty array for estimated model

print('Starting global search ...')

starttime = time.time()
for p in range(npos):
    model_est.append(GlobalSearch_3Lay(LUT, data[p], conds, thicks, nsl=len(conds)))
endtime = time.time() - starttime

print('Global search Q+IP excution for ', npos, ' positions: ', f"{(endtime/60):.3}", 'minutes')

# Save estimated models
np.save('results/model_3Lay_GS_field', model_est)
