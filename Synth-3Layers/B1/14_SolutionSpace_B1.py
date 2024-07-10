#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 14_SolutionSpace_B1.py
Description: Script to calculate solution space in B1 cases
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 17/12/2023
"""

# Import libraries
import numpy as np
from joblib import Parallel, delayed
import time
import sys
sys.path.insert(1, '../../src')

# Import functions
from EM1D import SolSpa_3Lay_parallel

# Load true data
data_B1_1 = np.load('data/data_synth_B1_1.npy')
data_B1_2 = np.load('data/data_synth_B1_2.npy')
data_B1_3 = np.load('data/data_synth_B1_3.npy')
data_B1_4 = np.load('data/data_synth_B1_4.npy')

# Lookup table ranges
conds = np.load('../data/conds.npy')
thicks = np.load('../data/thicks.npy')
nsl = len(conds)  # Lookupt table sampling number

# For example let's check the 1D model following position
pos = 10

# Analyze error space

n_workers = 48
max_err = 0.3 # Testing larger noise

print('Start calculating error B1-1 ...')
startTime = time.time()

models_err_B1_1 = Parallel(n_jobs=n_workers,verbose=0)(delayed(SolSpa_3Lay_parallel)(data_B1_1[pos], max_err, h1, h2, s1, s2, s3)
                                           for h1 in thicks for h2 in thicks for s1 in conds for s2 in conds for s3 in conds)

executionTime = ((time.time() - startTime))/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

mod_err = [i for i in models_err_B1_1 if i is not None]
models_err_B1_1_snip = np.array(mod_err)
err_B1_1_snip = models_err_B1_1_snip[:,-1]
models_err_B1_1_snip = models_err_B1_1_snip[:,:5]

np.save('results/models_err_B1_1_0.3', models_err_B1_1_snip)
np.save('results/err_B1_1_0.3', err_B1_1_snip)

print('Start calculating error B1-2 ...')
startTime = time.time()

models_err_B1_2 = Parallel(n_jobs=n_workers,verbose=0)(delayed(SolSpa_3Lay_parallel)(data_B1_2[pos], max_err, h1, h2, s1, s2, s3)
                                           for h1 in thicks for h2 in thicks for s1 in conds for s2 in conds for s3 in conds)

executionTime = ((time.time() - startTime))/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

mod_err = [i for i in models_err_B1_2 if i is not None]
models_err_B1_2_snip = np.array(mod_err)
err_B1_2_snip = models_err_B1_2_snip[:,-1]
models_err_B1_2_snip = models_err_B1_2_snip[:,:5]

np.save('results/models_err_B1_2_0.3', models_err_B1_2_snip)
np.save('results/err_B1_2_0.3', err_B1_2_snip)

print('Start calculating error B1-3 ...')
startTime = time.time()

models_err_B1_3 = Parallel(n_jobs=n_workers,verbose=0)(delayed(SolSpa_3Lay_parallel)(data_B1_3[pos], max_err, h1, h2, s1, s2, s3)
                                           for h1 in thicks for h2 in thicks for s1 in conds for s2 in conds for s3 in conds)

executionTime = ((time.time() - startTime))/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

mod_err = [i for i in models_err_B1_3 if i is not None]
models_err_B1_3_snip = np.array(mod_err)
err_B1_3_snip = models_err_B1_3_snip[:,-1]
models_err_B1_3_snip = models_err_B1_3_snip[:,:5]

np.save('results/models_err_B1_3_0.3', models_err_B1_3_snip)
np.save('results/err_B1_3_0.3', err_B1_3_snip)

print('Start calculating error B1-4 ...')
startTime = time.time()

models_err_B1_4 = Parallel(n_jobs=n_workers,verbose=0)(delayed(SolSpa_3Lay_parallel)(data_B1_4[pos], max_err, h1, h2, s1, s2, s3)
                                           for h1 in thicks for h2 in thicks for s1 in conds for s2 in conds for s3 in conds)

executionTime = ((time.time() - startTime))/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

mod_err = [i for i in models_err_B1_4 if i is not None]
models_err_B1_4_snip = np.array(mod_err)
err_B1_4_snip = models_err_B1_4_snip[:,-1]
models_err_B1_4_snip = models_err_B1_4_snip[:,:5]

np.save('results/models_err_B1_4_0.3', models_err_B1_4_snip)
np.save('results/err_B1_4_0.3', err_B1_4_snip)






