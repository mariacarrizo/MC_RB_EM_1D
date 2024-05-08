# Testing global search case A.1 with noisy data 

# Import libraries
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import function that performs global search in lookup table for 2-layered models
from EM1D import GlobalSearch_2Lay

# Load lookup table and sampling 
LUT = np.load('../data/LUTable_2Lay.npy')
conds = np.load('../data/conds.npy')
thicks =  np.load('../data/thicks.npy')

# Load true model and data
data_A2_1 = np.load('data/data_synth_2Lay_A2_1.npy')
data_A2_2 = np.load('data/data_synth_2Lay_A2_2.npy')
data_A2_3 = np.load('data/data_synth_2Lay_A2_3.npy')
data_A2_4 = np.load('data/data_synth_2Lay_A2_4.npy')

model_A2_1 = np.load('models/model_synth_2Lay_A2_1.npy')
model_A2_2 = np.load('models/model_synth_2Lay_A2_2.npy')
model_A2_3 = np.load('models/model_synth_2Lay_A2_3.npy')
model_A2_4 = np.load('models/model_synth_2Lay_A2_4.npy')

# number of 1D models
npos = len(data_A2_1)
nsl = len(conds)

# Creating model and data arrays
data_noise_2_A2_1 = data_A2_1.copy()
data_noise_5_A2_1 = data_A2_1.copy()
data_noise_10_A2_1 = data_A2_1.copy()

data_noise_2_A2_2 = data_A2_2.copy()
data_noise_5_A2_2 = data_A2_2.copy()
data_noise_10_A2_2 = data_A2_2.copy()

data_noise_2_A2_3 = data_A2_3.copy()
data_noise_5_A2_3 = data_A2_3.copy()
data_noise_10_A2_3 = data_A2_3.copy()

data_noise_2_A2_4 = data_A2_4.copy()
data_noise_5_A2_4 = data_A2_4.copy()
data_noise_10_A2_4 = data_A2_4.copy()

model_n2_A2_1 = np.zeros_like(model_A2_1)
model_n2_A2_2 = np.zeros_like(model_A2_1)
model_n2_A2_3 = np.zeros_like(model_A2_1)
model_n2_A2_4 = np.zeros_like(model_A2_1)

model_n5_A2_1 = np.zeros_like(model_A2_1)
model_n5_A2_2 = np.zeros_like(model_A2_1)
model_n5_A2_3 = np.zeros_like(model_A2_1)
model_n5_A2_4 = np.zeros_like(model_A2_1)

model_n10_A2_1 = np.zeros_like(model_A2_1)
model_n10_A2_2 = np.zeros_like(model_A2_1)
model_n10_A2_3 = np.zeros_like(model_A2_1)
model_n10_A2_4 = np.zeros_like(model_A2_1)

# Adding noise to data
# for noise 2.5 %

error_2 = 0.025
np.random.seed(13) 
data_noise_2_A2_1 *= np.random.randn(np.size(data_A2_1)).reshape(np.shape(data_A2_1))* error_2+1

np.random.seed(14) 
data_noise_2_A2_2 *= np.random.randn(np.size(data_A2_2)).reshape(np.shape(data_A2_2))* error_2+1

np.random.seed(15) 
data_noise_2_A2_3 *= np.random.randn(np.size(data_A2_3)).reshape(np.shape(data_A2_3))* error_2+1

np.random.seed(16) 
data_noise_2_A2_4 *= np.random.randn(np.size(data_A2_4)).reshape(np.shape(data_A2_4))* error_2+1


# Estimate with both Quadrature and In Phase

print('Estimating models noise 2.5% ...')
for p in range(npos):
    model_n2_A2_1[p] = GlobalSearch_2Lay(LUT, data_noise_2_A2_1[p], conds, thicks, nsl)
    model_n2_A2_2[p] = GlobalSearch_2Lay(LUT, data_noise_2_A2_2[p], conds, thicks, nsl)
    model_n2_A2_3[p] = GlobalSearch_2Lay(LUT, data_noise_2_A2_3[p], conds, thicks, nsl)
    model_n2_A2_4[p] = GlobalSearch_2Lay(LUT, data_noise_2_A2_4[p], conds, thicks, nsl)

print('Done!')
# for noise 5 %

error_5 = 0.05
np.random.seed(17) 
data_noise_5_A2_1 *= np.random.randn(np.size(data_A2_1)).reshape(np.shape(data_A2_1))* error_5+1

np.random.seed(18) 
data_noise_5_A2_2 *= np.random.randn(np.size(data_A2_2)).reshape(np.shape(data_A2_2))* error_5+1

np.random.seed(19) 
data_noise_5_A2_3 *= np.random.randn(np.size(data_A2_3)).reshape(np.shape(data_A2_3))* error_5+1

np.random.seed(20) 
data_noise_5_A2_4 *= np.random.randn(np.size(data_A2_4)).reshape(np.shape(data_A2_4))* error_5+1

print('Estimating models noise 5% ...')
for p in range(npos):
    model_n5_A2_1[p] = GlobalSearch_2Lay(LUT, data_noise_5_A2_1[p], conds, thicks, nsl)
    model_n5_A2_2[p] = GlobalSearch_2Lay(LUT, data_noise_5_A2_2[p], conds, thicks, nsl)
    model_n5_A2_3[p] = GlobalSearch_2Lay(LUT, data_noise_5_A2_3[p], conds, thicks, nsl)
    model_n5_A2_4[p] = GlobalSearch_2Lay(LUT, data_noise_5_A2_4[p], conds, thicks, nsl)
      
print('Done!')
      
# for noise 10%
error_10 = 0.1

np.random.seed(21)
data_noise_10_A2_1 *= np.random.randn(np.size(data_A2_1)).reshape(np.shape(data_A2_1))* error_10+1

np.random.seed(22)
data_noise_10_A2_2 *= np.random.randn(np.size(data_A2_2)).reshape(np.shape(data_A2_2))* error_10+1

np.random.seed(23)
data_noise_10_A2_3 *= np.random.randn(np.size(data_A2_3)).reshape(np.shape(data_A2_3))* error_10+1

np.random.seed(24)
data_noise_10_A2_4 *= np.random.randn(np.size(data_A2_4)).reshape(np.shape(data_A2_4))* error_10+1

print('Estimating models noise 10% ...')
for p in range(npos):
    model_n10_A2_1[p] = GlobalSearch_2Lay(LUT, data_noise_10_A2_1[p], conds, thicks, nsl)
    model_n10_A2_2[p] = GlobalSearch_2Lay(LUT, data_noise_10_A2_2[p], conds, thicks, nsl)
    model_n10_A2_3[p] = GlobalSearch_2Lay(LUT, data_noise_10_A2_3[p], conds, thicks, nsl)
    model_n10_A2_4[p] = GlobalSearch_2Lay(LUT, data_noise_10_A2_4[p], conds, thicks, nsl)

print('Done!')
    
# Save data with added noise
np.save('data/data_n2_A2_1', data_noise_2_A2_1)
np.save('data/data_n2_A2_2', data_noise_2_A2_2)
np.save('data/data_n2_A2_3', data_noise_2_A2_3)
np.save('data/data_n2_A2_4', data_noise_2_A2_4)

np.save('data/data_n5_A2_1', data_noise_5_A2_1)
np.save('data/data_n5_A2_2', data_noise_5_A2_2)
np.save('data/data_n5_A2_3', data_noise_5_A2_3)
np.save('data/data_n5_A2_4', data_noise_5_A2_4)

np.save('data/data_n10_A2_1', data_noise_10_A2_1)
np.save('data/data_n10_A2_2', data_noise_10_A2_2)
np.save('data/data_n10_A2_3', data_noise_10_A2_3)
np.save('data/data_n10_A2_4', data_noise_10_A2_4)
    
# Save estimates    
np.save('results/model_GS_n2_A2_1', model_n2_A2_1)
np.save('results/model_GS_n2_A2_2', model_n2_A2_2)
np.save('results/model_GS_n2_A2_3', model_n2_A2_3)
np.save('results/model_GS_n2_A2_4', model_n2_A2_4)

np.save('results/model_GS_n5_A2_1', model_n5_A2_1)
np.save('results/model_GS_n5_A2_2', model_n5_A2_2)
np.save('results/model_GS_n5_A2_3', model_n5_A2_3)
np.save('results/model_GS_n5_A2_4', model_n5_A2_4)

np.save('results/model_GS_n10_A2_1', model_n10_A2_1)
np.save('results/model_GS_n10_A2_2', model_n10_A2_2)
np.save('results/model_GS_n10_A2_3', model_n10_A2_3)
np.save('results/model_GS_n10_A2_4', model_n10_A2_4)