#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 6_GaussNewton_Noise_A1.py
Description: Script to test Gauss-Newton inversion in A1 cases with noisy data 
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 18/12/2023
"""

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward modelling class for 2-layered models
from EM1D import EMf_2Lay_Opt_HVP_1D

# Load survey parameters
survey = np.load('../data/survey_2Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Import true models and data 
model = np.load('models/model_synth_2Lay_A1_1.npy')
data = np.load('data/data_synth_2Lay_A1_1.npy')

# number of 1D models positions
npos = len(data) 

# Load data with added noise
data_n2_A1_1 = np.load('data/data_n2_A1_1.npy')
data_n2_A1_2 = np.load('data/data_n2_A1_2.npy')
data_n2_A1_3 = np.load('data/data_n2_A1_3.npy')
data_n2_A1_4 = np.load('data/data_n2_A1_4.npy')

data_n5_A1_1 = np.load('data/data_n5_A1_1.npy')
data_n5_A1_2 = np.load('data/data_n5_A1_2.npy')
data_n5_A1_3 = np.load('data/data_n5_A1_3.npy')
data_n5_A1_4 = np.load('data/data_n5_A1_4.npy')

data_n10_A1_1 = np.load('data/data_n10_A1_1.npy')
data_n10_A1_2 = np.load('data/data_n10_A1_2.npy')
data_n10_A1_3 = np.load('data/data_n10_A1_3.npy')
data_n10_A1_4 = np.load('data/data_n10_A1_4.npy')


# Define initial model [h_1, sigma_1, sigma_2] (sigmas in S/m)
m0 = [3, 500/1000, 500/1000]

# Define regularization parameter (alpha in Equation 5)
lam = 0

# relative error for inversion class
error = 1e-3 

# Defining inversion limits and transformations
transThk = pg.trans.TransLogLU(0.1,7)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

#%%
# Optimization for data [Q + IP] noise 2.5 %

## Case A1-1
print('Estimating model n2 A1-1')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n2_A1_1[0]) * error

# Empty array to store estimated model A1-1 noise 2.5%
model_GN_n2_A1_1 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_A1_1[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n2_A1_1[pos] = model_GN_pos
print('End')

## Case A1-2
print('Estimating model n2 A1-2')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n2_A1_2[0]) * error

# Empty array to store estimated model A1-2 noise 2.5%
model_GN_n2_A1_2 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_A1_2[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n2_A1_2[pos] = model_GN_pos
print('End')

## Case A1-3
print('Estimating model n2 A1-3')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n2_A1_3[0]) * error

# Empty array to store estimated model A1-3 noise 2.5%
model_GN_n2_A1_3 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_A1_3[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n2_A1_3[pos] = model_GN_pos
print('End')

## Case A1-4
print('Estimating model n2 A1-4')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n2_A1_4[0]) * error

# Empty array to store estimated model A1-4 noise 2.5%
model_GN_n2_A1_4 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_A1_4[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n2_A1_4[pos] = model_GN_pos
print('End')

## Optimization for data [Q + IP] noise 5 %

## Case A1-1
print('Estimating model n5 A1-1')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n5_A1_1[0]) * error

# Empty array to store estimated model A1-1 noise 5%
model_GN_n5_A1_1 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_A1_1[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n5_A1_1[pos] = model_GN_pos
print('End')

## Case A1-2
print('Estimating model n5 A1-2')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n5_A1_2[0]) * error

# Empty array to store estimated model A1-2 noise 5%
model_GN_n5_A1_2 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_A1_2[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n5_A1_2[pos] = model_GN_pos
print('End')

## Case A1-3
print('Estimating model n5 A1-3')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n5_A1_3[0]) * error

# Empty array to store estimated model A1-3 noise 5%
model_GN_n5_A1_3 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_A1_3[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n5_A1_3[pos] = model_GN_pos
print('End')

## Case A1-4
print('Estimating model n5 A1-4')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n5_A1_4[0]) * error

# Empty array to store estimated model A1-4 noise 5%
model_GN_n5_A1_4 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_A1_4[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n5_A1_4[pos] = model_GN_pos
print('End')

## Optimization for data [Q + IP] noise 10 %

## Case A1-1
print('Estimating model n10 A1-1')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n10_A1_1[0]) * error

# Empty array to store estimated model A1-1 noise 10%
model_GN_n10_A1_1 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_A1_1[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n10_A1_1[pos] = model_GN_pos
print('End')

## Case A1-2
print('Estimating model n10 A1-2')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n10_A1_2[0]) * error

# Empty array to store estimated model A1-2 noise 10%
model_GN_n10_A1_2 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_A1_2[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n10_A1_2[pos] = model_GN_pos
print('End')

## Case A1-3
print('Estimating model n10 A1-3')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n10_A1_3[0]) * error

# Empty array to store estimated model A1-3 noise 10%
model_GN_n10_A1_3 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_A1_3[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n10_A1_3[pos] = model_GN_pos
print('End')

## Case A1-4
print('Estimating model n10 A1-4')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n10_A1_4[0]) * error

# Empty array to store estimated model A1-4 noise 10%
model_GN_n10_A1_4 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_A1_4[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n10_A1_4[pos] = model_GN_pos
print('End')

# Store estimations
np.save('results/model_n2_GN_A1_1', model_GN_n2_A1_1)
np.save('results/model_n2_GN_A1_2', model_GN_n2_A1_2)
np.save('results/model_n2_GN_A1_3', model_GN_n2_A1_3)
np.save('results/model_n2_GN_A1_4', model_GN_n2_A1_4)

np.save('results/model_n5_GN_A1_1', model_GN_n5_A1_1)
np.save('results/model_n5_GN_A1_2', model_GN_n5_A1_2)
np.save('results/model_n5_GN_A1_3', model_GN_n5_A1_3)
np.save('results/model_n5_GN_A1_4', model_GN_n5_A1_4)

np.save('results/model_n10_GN_A1_1', model_GN_n10_A1_1)
np.save('results/model_n10_GN_A1_2', model_GN_n10_A1_2)
np.save('results/model_n10_GN_A1_3', model_GN_n10_A1_3)
np.save('results/model_n10_GN_A1_4', model_GN_n10_A1_4)