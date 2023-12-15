# Import libraries

import numpy as np
import time

import sys
sys.path.insert(1, '../../src')

# Load function that performs global search in lookup table
from EM1D import GlobalSearch

# Load lookup table and sampling 
LUT = np.load('../data/LUTable_2Lay.npy')
conds = np.load('../data/conds.npy')
thicks =  np.load('../data/thicks.npy')

# Load true data and true model
data = np.load('data/data_synth_2Lay_A1.npy')
model = np.load('data/model_synth_2Lay_A1.npy')

# number of 1D positions
npos = len(data)

# Estimate with both Quadrature and In Phase
model_est = np.zeros_like(model) # Empty array for estimated model

starttime = time.time()
for p in range(npos):
    model_est[p] = GlobalSearch(LUT, data[p], conds, thicks)
endtime = time.time() - starttime

print('Global search Q+IP excution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

# Estimate using only Quadrature
model_est_Q = np.zeros_like(model) # Empty array for estimated model

starttime = time.time()
for p in range(npos):
    model_est_Q[p] = GlobalSearch(LUT[:,:9], data[p,:9], conds, thicks)
endtime = time.time() - starttime

print('Global search Q excution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

# Estimate using only In Phase
model_est_IP = np.zeros_like(model) # Empty array for estimated model

starttime = time.time()
for p in range(npos):
    model_est_IP[p] = GlobalSearch(LUT[:,9:], data[p,9:], conds, thicks)
endtime = time.time() - starttime

print('Global search IP excution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

# Save estimated models

np.save('results/model_est_A1', model_est)
np.save('results/model_est_Q_A1', model_est_Q)
np.save('results/model_est_IP_A1', model_est_IP)
