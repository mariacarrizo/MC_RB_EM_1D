#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 4_GlobalSearch_B1_Q.py
Description: Script to perform Global Search in B1 cases using only Quadrature
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 16/12/2023
"""

## Import libraries
import numpy as np
import time
from joblib import Parallel, delayed
import sys
path = '../../src'
sys.path.insert(0, path)

# Import global search function for 3-layered models
from EM1D import GlobalSearch_3Lay

# Load survey information
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()

offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Number of cores used to perform global search
n_workers=4

# Load conductivities and thicknesses sampled
conds = np.load('../data/conds.npy')
thicks = np.load('../data/thicks.npy')
nsl = len(conds) # number of samples

# Load lookup table
LUT = np.load('../data/LUTable_3Lay.npy')

## Load true synthetic model and data
data_B1_1 = np.load('data/data_synth_B1_1.npy')
data_B1_2 = np.load('data/data_synth_B1_2.npy')
data_B1_3 = np.load('data/data_synth_B1_3.npy')
data_B1_4 = np.load('data/data_synth_B1_4.npy')

npos = len(data_B1_1) # number of 1D models

# Start inversion
print('Started global search for model B1-1 ...')
startTime = time.time()

model_GS_B1_1 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT[:,:9], data_B1_1[pos,:9],
                      conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

# Start inversion
print('Started global search for model B1-2 ...')
startTime = time.time()

model_GS_B1_2 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT[:,:9], data_B1_2[pos,:9],
                      conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

# Start inversion
print('Started global search for model B1-3 ...')
startTime = time.time()

model_GS_B1_3 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT[:,:9], data_B1_3[pos,:9],
                      conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

# Start inversion
print('Started global search for model B1-4 ...')
startTime = time.time()

model_GS_B1_4 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT[:,:9], data_B1_4[pos,:9],
                      conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

# Save estimated models
np.save('results/model_GS_Q_B1_1', model_GS_B1_1)
np.save('results/model_GS_Q_B1_2', model_GS_B1_2)
np.save('results/model_GS_Q_B1_3', model_GS_B1_3)
np.save('results/model_GS_Q_B1_4', model_GS_B1_4)

