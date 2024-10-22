#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 8_GlobalSearch_B1_n5.py
Description: Script to perform Global Search for B1 cases in 5% noise presence
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 17/12/2023
"""

## Import libraries
import numpy as np
import time
from joblib import Parallel, delayed
import sys
path = '../../src'
sys.path.insert(0, path)

# Import global search function
from EM1D import GlobalSearch_3Lay

# Number of cores used to perform global search
n_workers=5

# Load conductivities and thicknesses sampled
conds = np.load('../data/conds.npy')
thicks = np.load('../data/thicks.npy')
nsl = len(conds) # number of samples

# Load lookup table
LUT = np.load('../data/LUTable_3Lay.npy')

## Load data with noise included
data_n5_B1_1 = np.load('data/data_n5_B1_1.npy')
data_n5_B1_2 = np.load('data/data_n5_B1_2.npy')
data_n5_B1_3 = np.load('data/data_n5_B1_3.npy')
data_n5_B1_4 = np.load('data/data_n5_B1_4.npy')

npos = len(data_n5_B1_1) # number of 1D models

# Start inversion
print('Started global search B1-1 ...')
startTime = time.time()

model_n5_B1_1 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT,
                    data_n5_B1_1[pos], conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

print('Started global search B1-2 ...')
startTime = time.time()

model_n5_B1_2 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT,
                    data_n5_B1_2[pos], conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

print('Started global search B1-3 ...')
startTime = time.time()

model_n5_B1_3 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT,
                    data_n5_B1_3[pos], conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

print('Started global search B1-4 ...')
startTime = time.time()

model_n5_B1_4 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT,
                    data_n5_B1_4[pos], conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

# Save estimated model
np.save('results/model_GS_n5_B1_1', model_n5_B1_1)
np.save('results/model_GS_n5_B1_2', model_n5_B1_2)
np.save('results/model_GS_n5_B1_3', model_n5_B1_3)
np.save('results/model_GS_n5_B1_4', model_n5_B1_4)

