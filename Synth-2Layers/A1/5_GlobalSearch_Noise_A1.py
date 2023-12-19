# Testing global search with noise case A1

# Import libraries
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Load function that performs global search in lookup table
from EM1D import GlobalSearch

# Load lookup table and sampling 
LUT = np.load('../data/LUTable_2Lay.npy')
conds = np.load('../data/conds.npy')
thicks =  np.load('../data/thicks.npy')

# Load data
data = np.load('data/data_synth_2Lay_A1.npy')
npos = len(data)

# Load model
model = np.load('data/model_synth_2Lay_A1.npy')

# Adding noise to data
data_noise_2 = data.copy()
data_noise_5 = data.copy()
data_noise_10 = data.copy()

# for noise 2.5 %
np.random.seed(1) 
error_2 = 0.025
data_noise_2 *= np.random.randn(np.size(data)).reshape(np.shape(data))* error_2+1
model_n2 = np.zeros_like(model)

# Estimate with both Quadrature and In Phase
for p in range(npos):
    model_n2[p] = GlobalSearch(LUT, data_noise_2[p], conds, thicks)

# for noise 5 %
np.random.seed(2) 
error_5 = 0.05
data_noise_5 *= np.random.randn(np.size(data)).reshape(np.shape(data))* error_5+1
model_n5 = np.zeros_like(model)
for p in range(npos):
    model_n5[p] = GlobalSearch(LUT, data_noise_5[p], conds, thicks)

# for noise 10%
np.random.seed(3) 
error_10 = 0.1
data_noise_10 *= np.random.randn(np.size(data)).reshape(np.shape(data))* error_10+1
model_n10 = np.zeros_like(model)
for p in range(npos):
    model_n10[p] = GlobalSearch(LUT, data_noise_10[p], conds, thicks)
    
# Save data with added noise
np.save('data/data_A1_n2', data_noise_2)
np.save('data/data_A1_n5', data_noise_5)
np.save('data/data_A1_n10', data_noise_10)
    
# Save estimates    
np.save('results/model_2Lay_A1_GS_n2', model_n2)
np.save('results/model_2Lay_A1_GS_n5', model_n5)
np.save('results/model_2Lay_A1_GS_n10', model_n10)