""" 
Author: @mariacarrizo
Date created: 15/12/2024

Script to perform global search inversion for case A.1
"""

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
data_A1_1 = np.load('data/data_synth_2Lay_A1_1.npy')
data_A1_2 = np.load('data/data_synth_2Lay_A1_2.npy')
data_A1_3 = np.load('data/data_synth_2Lay_A1_3.npy')
data_A1_4 = np.load('data/data_synth_2Lay_A1_4.npy')

model_A1_1 = np.load('models/model_synth_2Lay_A1_1.npy')
model_A1_2 = np.load('models/model_synth_2Lay_A1_2.npy')
model_A1_3 = np.load('models/model_synth_2Lay_A1_3.npy')
model_A1_4 = np.load('models/model_synth_2Lay_A1_4.npy')

# number of 1D models
npos = len(data_A1_1)

# Estimate with both Quadrature and In Phase
model_GS_A1_1 = np.zeros_like(model_A1_1) # Empty array for estimated model
model_GS_A1_2 = np.zeros_like(model_A1_2)
model_GS_A1_3 = np.zeros_like(model_A1_3)
model_GS_A1_4 = np.zeros_like(model_A1_4)

print('Estimating model A1-1:')
starttime = time.time()
for p in range(npos):
    model_GS_A1_1[p] = GlobalSearch_2Lay(LUT, data_A1_1[p], conds, thicks)
endtime = time.time() - starttime
print('Global search Q+IP execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

print('Estimating model A1-2:')
starttime = time.time()
for p in range(npos):
    model_GS_A1_2[p] = GlobalSearch_2Lay(LUT, data_A1_2[p], conds, thicks)
endtime = time.time() - starttime
print('Global search Q+IP execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

print('Estimating model A1-3:')
starttime = time.time()
for p in range(npos):
    model_GS_A1_3[p] = GlobalSearch_2Lay(LUT, data_A1_3[p], conds, thicks)
endtime = time.time() - starttime
print('Global search Q+IP execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

print('Estimating model A1-4:')
starttime = time.time()
for p in range(npos):
    model_GS_A1_4[p] = GlobalSearch_2Lay(LUT, data_A1_4[p], conds, thicks)
endtime = time.time() - starttime
print('Global search Q+IP execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

# Estimate using only Quadrature
model_GS_Q_A1_1 = np.zeros_like(model_A1_1) # Empty array for estimated model
model_GS_Q_A1_2 = np.zeros_like(model_A1_2)
model_GS_Q_A1_3 = np.zeros_like(model_A1_3)
model_GS_Q_A1_4 = np.zeros_like(model_A1_4)

print('Estimating model A1-1:')
starttime = time.time()
for p in range(npos):
    model_GS_Q_A1_1[p] = GlobalSearch_2Lay(LUT[:,:9], data_A1_1[p,:9], conds, thicks)
endtime = time.time() - starttime
print('Global search Q execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

print('Estimating model A1-2:')
starttime = time.time()
for p in range(npos):
    model_GS_Q_A1_2[p] = GlobalSearch_2Lay(LUT[:,:9], data_A1_2[p,:9], conds, thicks)
endtime = time.time() - starttime
print('Global search Q execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

print('Estimating model A1-3:')
starttime = time.time()
for p in range(npos):
    model_GS_Q_A1_3[p] = GlobalSearch_2Lay(LUT[:,:9], data_A1_3[p,:9], conds, thicks)
endtime = time.time() - starttime
print('Global search Q execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

print('Estimating model A1-4:')
starttime = time.time()
for p in range(npos):
    model_GS_Q_A1_4[p] = GlobalSearch_2Lay(LUT[:,:9], data_A1_4[p,:9], conds, thicks)
endtime = time.time() - starttime
print('Global search Q execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

# Estimate using only In Phase
model_GS_IP_A1_1 = np.zeros_like(model_A1_1) # Empty array for estimated model
model_GS_IP_A1_2 = np.zeros_like(model_A1_2)
model_GS_IP_A1_3 = np.zeros_like(model_A1_3)
model_GS_IP_A1_4 = np.zeros_like(model_A1_4)

print('Estimating model A1-1:')
starttime = time.time()
for p in range(npos):
    model_GS_IP_A1_1[p] = GlobalSearch_2Lay(LUT[:,9:], data_A1_1[p,9:], conds, thicks)
endtime = time.time() - starttime
print('Global search IP execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

print('Estimating model A1-2:')
starttime = time.time()
for p in range(npos):
    model_GS_IP_A1_2[p] = GlobalSearch_2Lay(LUT[:,9:], data_A1_2[p,9:], conds, thicks)
endtime = time.time() - starttime
print('Global search IP execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

print('Estimating model A1-3:')
starttime = time.time()
for p in range(npos):
    model_GS_IP_A1_3[p] = GlobalSearch_2Lay(LUT[:,9:], data_A1_3[p,9:], conds, thicks)
endtime = time.time() - starttime
print('Global search IP execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

print('Estimating model A1-4:')
starttime = time.time()
for p in range(npos):
    model_GS_IP_A1_4[p] = GlobalSearch_2Lay(LUT[:,9:], data_A1_4[p,9:], conds, thicks)
endtime = time.time() - starttime
print('Global search IP execution for ', npos, ' positions: ', f"{endtime:.3}", ' seconds')

# Save estimated models
np.save('results/model_2Lay_A1_1', model_GS_A1_1)
np.save('results/model_2Lay_A1_2', model_GS_A1_2)
np.save('results/model_2Lay_A1_3', model_GS_A1_3)
np.save('results/model_2Lay_A1_4', model_GS_A1_4)

np.save('results/model_2Lay_Q_A1_1', model_GS_Q_A1_1)
np.save('results/model_2Lay_Q_A1_2', model_GS_Q_A1_2)
np.save('results/model_2Lay_Q_A1_3', model_GS_Q_A1_3)
np.save('results/model_2Lay_Q_A1_4', model_GS_Q_A1_4)

np.save('results/model_2Lay_IP_A1_1', model_GS_IP_A1_1)
np.save('results/model_2Lay_IP_A1_2', model_GS_IP_A1_2)
np.save('results/model_2Lay_IP_A1_3', model_GS_IP_A1_3)
np.save('results/model_2Lay_IP_A1_4', model_GS_IP_A1_4)
