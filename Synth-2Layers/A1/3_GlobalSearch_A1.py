# Script to perform global search inversion for case A.1

# Import libraries

import numpy as np
import time
import sys
sys.path.insert(1, '../../src')

# Import function that performs global search in lookup table
from EM1D import GlobalSearch_2Lay

# Load lookup table and sampling 
LUT = np.load('../data/LUTable_2Lay.npy')
conds = np.load('../data/conds.npy')
thicks =  np.load('../data/thicks.npy')

# Load true data and true model
data = np.load('data/data_synth_2Lay_A1.npy')
model = np.load('data/model_synth_2Lay_A1.npy')

# number of 1D models
npos = len(data)

# Estimate with both Quadrature and In Phase
model_GS = np.zeros_like(model) # Empty array for estimated model

starttime = time.time()
for p in range(npos):
    model_GS[p] = GlobalSearch_2Lay(LUT, data[p], conds, thicks)
endtime = time.time() - starttime

print('Global search Q+IP excution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

# Estimate using only Quadrature
model_GS_Q = np.zeros_like(model) # Empty array for estimated model

starttime = time.time()
for p in range(npos):
    model_GS_Q[p] = GlobalSearch_2Lay(LUT[:,:9], data[p,:9], conds, thicks)
endtime = time.time() - starttime

print('Global search Q excution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

# Estimate using only In Phase
model_GS_IP = np.zeros_like(model) # Empty array for estimated model

starttime = time.time()
for p in range(npos):
    model_GS_IP[p] = GlobalSearch_2Lay(LUT[:,9:], data[p,9:], conds, thicks)
endtime = time.time() - starttime

print('Global search IP excution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

# Save estimated models
np.save('results/model_2Lay_A1', model_GS)
np.save('results/model_2Lay_A1_Q', model_GS_Q)
np.save('results/model_2Lay_A1_IP', model_GS_IP)
