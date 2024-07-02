#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 5_GlobalSearch_Noise_A1.py
Description: Script to test global search in A1 cases with noisy data 
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 18/12/2023
"""

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
data_A1_1 = np.load('data/data_synth_2Lay_A1_1.npy')
data_A1_2 = np.load('data/data_synth_2Lay_A1_2.npy')
data_A1_3 = np.load('data/data_synth_2Lay_A1_3.npy')
data_A1_4 = np.load('data/data_synth_2Lay_A1_4.npy')

model_A1_1 = np.load('models/model_synth_2Lay_A1_1.npy')
model_A1_2 = np.load('models/model_synth_2Lay_A1_2.npy')
model_A1_3 = np.load('models/model_synth_2Lay_A1_3.npy')
model_A1_4 = np.load('models/model_synth_2Lay_A1_4.npy')

# number of 1D models positions
npos = len(data_A1_1)

# sampling number in lookup table
nsl = len(conds)

# Creating model and data arrays
data_noise_2_A1_1 = data_A1_1.copy()
data_noise_5_A1_1 = data_A1_1.copy()
data_noise_10_A1_1 = data_A1_1.copy()

data_noise_2_A1_2 = data_A1_2.copy()
data_noise_5_A1_2 = data_A1_2.copy()
data_noise_10_A1_2 = data_A1_2.copy()

data_noise_2_A1_3 = data_A1_3.copy()
data_noise_5_A1_3 = data_A1_3.copy()
data_noise_10_A1_3 = data_A1_3.copy()

data_noise_2_A1_4 = data_A1_4.copy()
data_noise_5_A1_4 = data_A1_4.copy()
data_noise_10_A1_4 = data_A1_4.copy()

model_n2_A1_1 = np.zeros_like(model_A1_1)
model_n2_A1_2 = np.zeros_like(model_A1_1)
model_n2_A1_3 = np.zeros_like(model_A1_1)
model_n2_A1_4 = np.zeros_like(model_A1_1)

model_n5_A1_1 = np.zeros_like(model_A1_1)
model_n5_A1_2 = np.zeros_like(model_A1_1)
model_n5_A1_3 = np.zeros_like(model_A1_1)
model_n5_A1_4 = np.zeros_like(model_A1_1)

model_n10_A1_1 = np.zeros_like(model_A1_1)
model_n10_A1_2 = np.zeros_like(model_A1_1)
model_n10_A1_3 = np.zeros_like(model_A1_1)
model_n10_A1_4 = np.zeros_like(model_A1_1)

# Adding noise to data
# for noise 2.5 %

error_2 = 0.025
np.random.seed(1) 
data_noise_2_A1_1 *= np.random.randn(np.size(data_A1_1)).reshape(np.shape(data_A1_1))* error_2+1

np.random.seed(2) 
data_noise_2_A1_2 *= np.random.randn(np.size(data_A1_2)).reshape(np.shape(data_A1_2))* error_2+1

np.random.seed(3) 
data_noise_2_A1_3 *= np.random.randn(np.size(data_A1_3)).reshape(np.shape(data_A1_3))* error_2+1

np.random.seed(4) 
data_noise_2_A1_4 *= np.random.randn(np.size(data_A1_4)).reshape(np.shape(data_A1_4))* error_2+1


# Estimate with both Quadrature and In Phase

print('Estimating models noise 2.5% ...')
for p in range(npos):
    model_n2_A1_1[p] = GlobalSearch_2Lay(LUT, data_noise_2_A1_1[p], conds, thicks, nsl)
    model_n2_A1_2[p] = GlobalSearch_2Lay(LUT, data_noise_2_A1_2[p], conds, thicks, nsl)
    model_n2_A1_3[p] = GlobalSearch_2Lay(LUT, data_noise_2_A1_3[p], conds, thicks, nsl)
    model_n2_A1_4[p] = GlobalSearch_2Lay(LUT, data_noise_2_A1_4[p], conds, thicks, nsl)

print('Done!')
# for noise 5 %

error_5 = 0.05
np.random.seed(5) 
data_noise_5_A1_1 *= np.random.randn(np.size(data_A1_1)).reshape(np.shape(data_A1_1))* error_5+1

np.random.seed(6) 
data_noise_5_A1_2 *= np.random.randn(np.size(data_A1_2)).reshape(np.shape(data_A1_2))* error_5+1

np.random.seed(7) 
data_noise_5_A1_3 *= np.random.randn(np.size(data_A1_3)).reshape(np.shape(data_A1_3))* error_5+1

np.random.seed(8) 
data_noise_5_A1_4 *= np.random.randn(np.size(data_A1_4)).reshape(np.shape(data_A1_4))* error_5+1

print('Estimating models noise 5% ...')
for p in range(npos):
    model_n5_A1_1[p] = GlobalSearch_2Lay(LUT, data_noise_5_A1_1[p], conds, thicks, nsl)
    model_n5_A1_2[p] = GlobalSearch_2Lay(LUT, data_noise_5_A1_2[p], conds, thicks, nsl)
    model_n5_A1_3[p] = GlobalSearch_2Lay(LUT, data_noise_5_A1_3[p], conds, thicks, nsl)
    model_n5_A1_4[p] = GlobalSearch_2Lay(LUT, data_noise_5_A1_4[p], conds, thicks, nsl)
      
print('Done!')
      
# for noise 10%
error_10 = 0.1

np.random.seed(9)
data_noise_10_A1_1 *= np.random.randn(np.size(data_A1_1)).reshape(np.shape(data_A1_1))* error_10+1

np.random.seed(10)
data_noise_10_A1_2 *= np.random.randn(np.size(data_A1_2)).reshape(np.shape(data_A1_2))* error_10+1

np.random.seed(11)
data_noise_10_A1_3 *= np.random.randn(np.size(data_A1_3)).reshape(np.shape(data_A1_3))* error_10+1

np.random.seed(12)
data_noise_10_A1_4 *= np.random.randn(np.size(data_A1_4)).reshape(np.shape(data_A1_4))* error_10+1

print('Estimating models noise 10% ...')
for p in range(npos):
    model_n10_A1_1[p] = GlobalSearch_2Lay(LUT, data_noise_10_A1_1[p], conds, thicks, nsl)
    model_n10_A1_2[p] = GlobalSearch_2Lay(LUT, data_noise_10_A1_2[p], conds, thicks, nsl)
    model_n10_A1_3[p] = GlobalSearch_2Lay(LUT, data_noise_10_A1_3[p], conds, thicks, nsl)
    model_n10_A1_4[p] = GlobalSearch_2Lay(LUT, data_noise_10_A1_4[p], conds, thicks, nsl)

print('Done!')
    
# Save data with added noise
np.save('data/data_n2_A1_1', data_noise_2_A1_1)
np.save('data/data_n2_A1_2', data_noise_2_A1_2)
np.save('data/data_n2_A1_3', data_noise_2_A1_3)
np.save('data/data_n2_A1_4', data_noise_2_A1_4)

np.save('data/data_n5_A1_1', data_noise_5_A1_1)
np.save('data/data_n5_A1_2', data_noise_5_A1_2)
np.save('data/data_n5_A1_3', data_noise_5_A1_3)
np.save('data/data_n5_A1_4', data_noise_5_A1_4)

np.save('data/data_n10_A1_1', data_noise_10_A1_1)
np.save('data/data_n10_A1_2', data_noise_10_A1_2)
np.save('data/data_n10_A1_3', data_noise_10_A1_3)
np.save('data/data_n10_A1_4', data_noise_10_A1_4)
    
# Save estimates    
np.save('results/model_GS_n2_A1_1', model_n2_A1_1)
np.save('results/model_GS_n2_A1_2', model_n2_A1_2)
np.save('results/model_GS_n2_A1_3', model_n2_A1_3)
np.save('results/model_GS_n2_A1_4', model_n2_A1_4)

np.save('results/model_GS_n5_A1_1', model_n5_A1_1)
np.save('results/model_GS_n5_A1_2', model_n5_A1_2)
np.save('results/model_GS_n5_A1_3', model_n5_A1_3)
np.save('results/model_GS_n5_A1_4', model_n5_A1_4)

np.save('results/model_GS_n10_A1_1', model_n10_A1_1)
np.save('results/model_GS_n10_A1_2', model_n10_A1_2)
np.save('results/model_GS_n10_A1_3', model_n10_A1_3)
np.save('results/model_GS_n10_A1_4', model_n10_A1_4)